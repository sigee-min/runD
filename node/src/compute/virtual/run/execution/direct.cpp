#include "../execution.hpp"

#include "../backing.hpp"
#include "../cache.hpp"
#include "fill.hpp"

#include "../../../../hash/fnv.hpp"
#include "../../../backend.hpp"
#include "../../../device/residency/execution/run.hpp"
#include "../../../device/residency/execution/sliding.hpp"
#include "../../../device/residency/execution/stream.hpp"
#include "../../../device/residency/pool.hpp"
#include "../../../pipeline/execution/attempt.hpp"
#include "../../../pipeline/execution/schedule.hpp"
#include "../../../pipeline/execution/submit.hpp"
#include "../../../pipeline/execution/window.hpp"
#include "../../../pipeline/local.hpp"
#include "../../../pipeline/run/clock.hpp"
#include "../../../pipeline/state.hpp"
#include "../../../pipeline/transfer/batch.hpp"
#include "../../backing.hpp"
#include "../../stats.hpp"

#include <rund/compute/pipeline/runtime.hpp>
#include <rund/counter.hpp>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <span>
#include <thread>

#include "io.hpp"

namespace rund::compute::detail {
using ::rund::detail::counter::Accumulate;

namespace {
struct NativeWait final {
  std::mutex gate{};
  std::condition_variable ready{};
  residency::execution::NativeEvidence evidence{};
  bool completed{};
};

void complete_native(void *const raw,
                     residency::execution::NativeEvidence &&evidence) noexcept {
  auto *const wait = static_cast<NativeWait *>(raw);
  if (wait == nullptr) {
    return;
  }
  {
    std::lock_guard lock{wait->gate};
    if (wait->completed) {
      return;
    }
    wait->evidence = std::move(evidence);
    wait->completed = true;
    wait->ready.notify_one();
  }
}

} // namespace

Status prepare_virtual_execution(VirtualPipelineState &state,
                                 const VirtualRunProjection &run,
                                 VirtualExecutionPrepared &prepared) noexcept {
  prepared = {};
  const residency::execution::SealResult sealed =
      seal_virtual_execution(state, run);
  // The retained Q1 native owner is a pointwise-only contract. A one-epoch
  // Window still needs selected-local range semantics, so it remains on the
  // exact rolling path instead of entering a native owner that cannot prove
  // that topology. Q>=2 Window uses the recurrent window owner below.
  if (state.geometry.route != VirtualRoute::Pointwise || !sealed ||
      sealed.plan.epoch_count() != 1u || state.pipeline == nullptr ||
      state.pipeline->device == nullptr ||
      state.pipeline->device->ops == nullptr ||
      state.pipeline->device->ops->residency.prepare_residency_execution ==
          nullptr) {
    return Status::fail(Reason::BackendUnsupported);
  }
  const node::accel::detail::PreparedKernelPipeline *native = nullptr;
  {
    std::lock_guard pipeline_lock{state.pipeline->gate};
    std::lock_guard publication_lock{state.pipeline->publication->gate};
    native = state.pipeline->transactional &&
                     state.pipeline->publication->parity != 0u
                 ? &state.pipeline->alternate_prepared
                 : &state.pipeline->prepared;
  }
  residency::execution::Owner owner{};
  const Status status =
      state.pipeline->device->ops->residency.prepare_residency_execution(
          *state.pipeline->device,
          std::span<const node::accel::detail::PreparedKernelPipeline *const>{
              &native, 1u},
          owner);
  if (!status || !owner) {
    return status ? Status::fail(Reason::BackendUnsupported) : status;
  }
  prepared.plan = sealed.plan;
  prepared.owner = std::move(owner);
  return Status::success();
}

VirtualExecutionResult execute_virtual_execution(
    VirtualPipelineState &state, VirtualBacking &input, VirtualBacking &output,
    const VirtualRunProjection &run, const VirtualExecutionPrepared &prepared,
    Stats &stats) noexcept {
  using namespace residency;
  using namespace residency::execution;
  VirtualExecutionResult result{};
  PipelineState &pipeline = *state.pipeline;
  Pool &pool = *pipeline.residency_pool;
  Run joined{};
  const ExecutionLease lease = joined.begin(pool.authority(), prepared.plan);
  if (!lease) {
    result.status = Status::fail(lease.failure == AuthorityFailure::Busy
                                     ? Reason::PipelineBusy
                                     : Reason::PipelineMemoryBudget);
    return result;
  }
  const auto internal_failure = [&](const std::uint64_t page,
                                    const bool native_observed =
                                        false) noexcept {
    const bool abandoned = joined.abandon();
    result.status = Status::fail(Reason::PipelineInvalid);
    result.failed_page = page;
    result.poison_pipeline = true;
    if (native_observed) {
      static_cast<void>(
          accumulate_virtual_epoch(stats, pipeline_stats(state.pipeline)));
    }
    // A failed abandon is itself an Authority contradiction. The Pipeline is
    // already poisoned, so no caller can reinterpret this as retry authority.
    static_cast<void>(abandoned);
    return result;
  };
  ExecutionTicket input_ticket{};
  if (!joined.issue(Phase::Input, input_ticket)) {
    return internal_failure(ResidencyStats::no_failed_page);
  }

  std::uint64_t backing_pages = 0u;
  Status status =
      execution_io::fill_input(input, run, input_ticket, &pipeline,
                               stats.pipeline.residency, backing_pages);
  std::uint64_t device_pages = 0u;
  if (status) {
    status = execution_io::supply_input(pipeline, run, input_ticket, stats,
                                        device_pages);
  }
  if (!status) {
    const bool terminal =
        joined.terminal(input_ticket, ExecutionTerminal::Failure, status);
    const ExecutionClose closed = joined.close(pipeline_clock());
    if (!terminal || !closed) {
      return internal_failure(execution_io::failed_page(input_ticket));
    }
    result.status = status;
    result.failed_page = execution_io::failed_page(input_ticket);
    return result;
  }
  if (!joined.terminal(input_ticket, ExecutionTerminal::Success,
                       Status::success())) {
    return internal_failure(execution_io::failed_page(input_ticket));
  }

  const std::size_t count = input_ticket.bindings.size() / 2u;
  Accumulate(stats.pipeline.residency.page_in_count, device_pages);
  Accumulate(stats.pipeline.residency.cache_hit_count, count - device_pages);
  Accumulate(stats.pipeline.residency.page_in_bytes,
             device_pages * run.input_page_bytes);
  Accumulate(stats.pipeline.residency.late_page_count, backing_pages);

  NativeWait wait{};
  NativeEvidence native{};
  Control control{
      .token = lease.token,
      .generation = lease.generation,
      .evidence = &native,
      .completion = complete_native,
      .user = &wait,
  };
  PipelineExecutionSubmission submission{};
  const std::uint64_t submitted_ns = pipeline_clock();
  status = submit_pipeline_execution(state.pipeline, prepared.owner,
                                     prepared.plan, control, submission);
  if (!status) {
    const bool rejected = joined.reject(status);
    const ExecutionClose closed = joined.close(pipeline_clock());
    if (!rejected || !closed) {
      return internal_failure(execution_io::failed_page(input_ticket));
    }
    result.status = status;
    result.failed_page = execution_io::failed_page(input_ticket);
    return result;
  }
  {
    std::unique_lock lock{wait.gate};
    wait.ready.wait(lock, [&wait] { return wait.completed; });
    native = wait.evidence;
  }
  Accumulate(stats.pipeline.residency.stall_ns,
             pipeline_clock() - submitted_ns);
  submission.reset();
  if (!joined.native(native)) {
    return internal_failure(execution_io::failed_page(input_ticket), true);
  }
  if (!native.status) {
    const ExecutionClose closed = joined.close(pipeline_clock());
    if (!closed) {
      return internal_failure(execution_io::failed_page(input_ticket), true);
    }
    result.status = native.status;
    result.failed_page = execution_io::failed_page(input_ticket);
    result.poison_pipeline = native.terminal == TerminalKind::UnknownMayWrite ||
                             poisoned_pipeline(state.pipeline);
    static_cast<void>(
        accumulate_virtual_epoch(stats, pipeline_stats(state.pipeline)));
    return result;
  }

  ExecutionTicket output_ticket{};
  if (!joined.issue(Phase::Output, output_ticket)) {
    return internal_failure(execution_io::failed_page(input_ticket), true);
  }
  std::array<const std::byte *, PipelineLeafCapacity> output_storage{};
  std::span<const std::byte *const> output_frames{};
  status = execution_io::collect_output(pipeline, run, output_ticket,
                                        output_frames, output_storage);
  if (status) {
    status =
        stage_residency_cache(output, run, output_ticket.transitions,
                              stats.pipeline.residency, nullptr, output_frames);
  }
  if (!status) {
    const bool terminal =
        joined.terminal(output_ticket, ExecutionTerminal::Failure, status);
    const ExecutionClose closed = joined.close(pipeline_clock());
    if (!terminal || !closed) {
      return internal_failure(execution_io::failed_page(output_ticket), true);
    }
    result.status = status;
    result.failed_page = execution_io::failed_page(output_ticket);
    static_cast<void>(
        accumulate_virtual_epoch(stats, pipeline_stats(state.pipeline)));
    return result;
  }
  if (!joined.terminal(output_ticket, ExecutionTerminal::Success,
                       Status::success())) {
    return internal_failure(execution_io::failed_page(output_ticket), true);
  }
  if (consume_virtual_execution_close_failure_once()) {
    return internal_failure(execution_io::failed_page(output_ticket), true);
  }
  const ExecutionClose closed = joined.close(pipeline_clock());
  if (!closed || !closed.success) {
    return internal_failure(execution_io::failed_page(output_ticket), true);
  }
  publish_residency_cache(output);
  clear_virtual_recovery(output);

  ::rund::node::hash_detail::Fnv hash{};
  for (std::size_t local = 0u; local < output_frames.size(); ++local) {
    const CacheBinding source = output_ticket.bindings[local * 2u];
    const std::uint64_t logical = source.key.page * run.output_payload_bytes;
    const std::size_t bytes = static_cast<std::size_t>(
        std::min(run.output_payload_bytes, run.active.output_bytes - logical));
    hash.Bytes(reinterpret_cast<const std::uint8_t *>(output_frames[local] +
                                                      run.output_prefix_bytes),
               bytes);
  }
  Accumulate(stats.pipeline.residency.epoch_count, 1u);
  const Status folded =
      accumulate_virtual_epoch(stats, pipeline_stats(state.pipeline));
  result.status = folded;
  result.output_hash = folded ? hash.Finish() : 0u;
  result.poison_pipeline = !folded;
  return result;
}

} // namespace rund::compute::detail
