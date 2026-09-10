#include "internal.hpp"

#include "../../../stats.hpp"
#include "../../cache.hpp"

#include "../../../../backend.hpp"
#include "../../../../device/residency/pool.hpp"
#include "../../../../pipeline/run/clock.hpp"
#include "../../../../pipeline/state.hpp"

#include <rund/counter.hpp>

#include <array>
#include <span>
#include <thread>

namespace rund::compute::detail {
using ::rund::detail::counter::Accumulate;

[[nodiscard]] VirtualExecutionResult execute_virtual_execution_stream(
    VirtualPipelineState &state, VirtualBacking &input, VirtualBacking &output,
    const VirtualRunProjection &run,
    const VirtualExecutionWindowPrepared &prepared, Stats &stats) noexcept {
  using namespace residency;
  using namespace residency::execution;
  using namespace virtual_stream_detail;
  VirtualExecutionResult result{};
  if (!prepared || prepared.plan.epoch_count() <= WindowCapacity ||
      state.pipeline == nullptr || state.alternate_pipeline == nullptr ||
      state.pipeline->residency_pool == nullptr ||
      state.pipeline->device == nullptr ||
      state.pipeline->device->ops == nullptr ||
      state.pipeline->device->ops->residency.claim_residency_stream ==
          nullptr ||
      state.pipeline->device->ops->residency.release_residency_stream ==
          nullptr ||
      state.pipeline->device->ops->residency.quarantine_residency_stream ==
          nullptr ||
      state.pipeline->device->ops->residency.submit_residency_stream_window ==
          nullptr) {
    return result;
  }
  PipelineExecutionSchedulePrepared native_schedule{};
  const Status schedule_status = prepare_pipeline_execution_schedule(
      prepared.plan, prepared.pipelines, native_schedule);
  const bool scheduled = static_cast<bool>(schedule_status);
  Pool &pool = *state.pipeline->residency_pool;
  Stream joined{};
  const ExecutionLease lease =
      scheduled ? joined.begin_schedule(pool.authority(), prepared.plan)
                : joined.begin(pool.authority(), prepared.plan);
  if (!lease) {
    result.status = Status::fail(lease.failure == AuthorityFailure::Busy
                                     ? Reason::PipelineBusy
                                     : Reason::PipelineMemoryBudget);
    return result;
  }

  // Retain every native owner reachable by either bank's parity for the whole
  // recurrent run. Intermediate bounded Finals release only their window
  // pointer; an ordinary peer submission remains Busy until this sole stream
  // terminal releases the sentinel.
  std::array<node::accel::detail::PreparedKernelPipeline,
             node::accel::detail::ResidencyWindowCapacity>
      native_owners{};
  std::size_t native_count = 0u;
  for (const std::shared_ptr<PipelineState> &pipeline : prepared.pipelines) {
    if (pipeline == nullptr) {
      static_cast<void>(joined.abandon());
      return result;
    }
    std::lock_guard pipeline_lock{pipeline->gate};
    native_owners[native_count++] = pipeline->prepared;
    if (pipeline->transactional) {
      native_owners[native_count++] = pipeline->alternate_prepared;
    }
  }
  node::accel::detail::PreparedResidencyStreamControl claim{};
  Status status = state.pipeline->device->ops->residency.claim_residency_stream(
      *state.pipeline->device,
      std::span<const node::accel::detail::PreparedKernelPipeline>{
          native_owners.data(), native_count},
      lease.plan, lease.token, lease.generation, claim);
  if (!status) {
    const bool abandoned = joined.abandon();
    result.status = status;
    result.poison_pipeline = !abandoned;
    return result;
  }

  PipelineResidencyWindowControl native{};
  PipelineResidencyScheduleControl schedule_native{};
  StreamWait wait{
      .stream = &joined,
      .plan = &prepared.plan,
      .native = scheduled ? nullptr : &native,
      .schedule = scheduled ? &schedule_native : nullptr,
      .claim = &claim,
      .pool = &pool,
      .state = &state,
      .input = &input,
      .output = &output,
      .run = &run,
      .stats = &stats,
      .pipelines = prepared.pipelines,
      .chunk_count =
          scheduled ? static_cast<std::size_t>(lease.epochs) : WindowCapacity,
      .caller = std::this_thread::get_id(),
  };
  std::array<ExecutionTicket, BankCapacity> bootstrap_tickets{};
  const std::uint64_t bootstrap = BankCapacity;
  for (std::uint64_t epoch = 0u; epoch < bootstrap; ++epoch) {
    if (!joined.issue(epoch, Phase::Input,
                      bootstrap_tickets[static_cast<std::size_t>(epoch)])) {
      status = Status::fail(Reason::BackendUnsupported);
      break;
    }
  }
  const std::uint64_t submitted_ns = pipeline_clock();
  if (status) {
    status =
        scheduled
            ? submit_pipeline_execution_schedule(
                  prepared.plan, native_schedule, lease, complete_stream_native,
                  complete_stream_schedule_final, &wait, schedule_native, claim)
            : submit_pipeline_execution_stream_chunk(
                  prepared.plan, lease, 0u, WindowCapacity, prepared.pipelines,
                  complete_stream_native, complete_stream_final, &wait, native,
                  claim);
  }
  if (!status) {
    const bool abandoned = joined.abandon();
    const Status released =
        state.pipeline->device->ops->residency.release_residency_stream(
            *state.pipeline->device, claim, lease.plan, lease.token,
            lease.generation, !abandoned);
    const Status detached =
        released
            ? Status::success()
            : state.pipeline->device->ops->residency
                  .quarantine_residency_stream(*state.pipeline->device, claim,
                                               lease.plan, lease.token,
                                               lease.generation);
    result.status = status;
    result.poison_pipeline = !abandoned || !released || !detached;
    return result;
  }

  // Only the fixed two-bank bootstrap executes on the caller. Every later
  // chunk is enqueued by its preceding raw Final onto Pool's accelerator-only
  // cold serial lane; the caller performs one terminal wait below.
  for (std::uint64_t epoch = 0u; epoch < bootstrap; ++epoch) {
    Status admission = Status::success();
    {
      std::lock_guard lock{wait.coordinator};
      const ExecutionTicket &ticket =
          bootstrap_tickets[static_cast<std::size_t>(epoch)];
      if (!wait.disposition) {
        // The native chunk is already queued, but a prior bootstrap bank
        // failed before its ready signal. Terminal the remaining issued Host
        // service tickets without performing more backing I/O, then open
        // their native gates through Suppress. A later bootstrap bank must not
        // restart the run after the first exact failure.
        admission = wait.disposition;
        if (!joined.terminal(ticket, ExecutionTerminal::Failure, admission)) {
          admission = Status::fail(Reason::PipelineInvalid);
          wait.quarantine = true;
        }
      } else {
        admission = service_stream_input(wait, epoch, &ticket);
      }
      if (!admission && wait.disposition) {
        wait.disposition = admission;
      }
    }
    const Status signalled = signal_stream_native(wait, epoch, admission);
    if (!signalled) {
      {
        std::lock_guard lock{wait.coordinator};
        wait.disposition = signalled;
        wait.quarantine = true;
      }
      const Status aborted = abort_stream_native(wait, signalled);
      if (!aborted) {
        std::lock_guard lock{wait.coordinator};
        wait.disposition = aborted;
      }
      break;
    }
  }

  {
    std::unique_lock lock{wait.gate};
    wait.ready.wait(lock, [&wait] { return wait.completed; });
  }
  // Quiesce the cold continuation before releasing stack-owned callback
  // controls. This is a no-op when the terminal came directly from a backend
  // completion lane rather than the recurrent service.
  pool.wait_recurrent();
  Accumulate(stats.pipeline.residency.stall_ns,
             pipeline_clock() - submitted_ns);

  {
    std::lock_guard lock{wait.coordinator};
    if (wait.callback_on_caller && wait.disposition) {
      wait.disposition = Status::fail(Reason::CompletionInvalid);
    }
  }
  const bool quarantine = wait.poison || wait.callback_on_caller ||
                          wait.final.terminal == TerminalKind::UnknownMayWrite;
  const Status released =
      state.pipeline->device->ops->residency.release_residency_stream(
          *state.pipeline->device, claim, lease.plan, lease.token,
          lease.generation, quarantine);
  const Status detached =
      released
          ? Status::success()
          : state.pipeline->device->ops->residency.quarantine_residency_stream(
                *state.pipeline->device, claim, lease.plan, lease.token,
                lease.generation);
  const ExecutionClose closed = joined.close();
  if (!released || !detached || !closed) {
    result.status = Status::fail(!released ? Reason::BackendFailed
                                           : Reason::PipelineInvalid);
    result.poison_pipeline = true;
    return result;
  }
  Accumulate(stats.pipeline.residency.window_handoff_count,
             wait.final.public_handoffs);
  Accumulate(stats.pipeline.residency.window_batch_count,
             wait.final.native_batches);
  Accumulate(stats.pipeline.residency.window_queue_call_count,
             wait.final.queue_calls);
  stats.command_submits = wait.final.queue_calls;
  if (!closed.success || !wait.failure || !wait.final.status ||
      wait.callback_on_caller) {
    result.status = wait.callback_on_caller
                        ? Status::fail(Reason::CompletionInvalid)
                        : (!wait.failure ? wait.failure : wait.final.status);
    result.failed_page = wait.failed_page;
    // Authority conservatively invalidates every exact may-write region for
    // a Known failure. Only an Unknown terminal (or callback-integrity loss)
    // poisons the backend owner; a Known invalidation remains retryable.
    result.poison_pipeline = quarantine;
    return result;
  }
  publish_residency_cache(output);
  clear_virtual_recovery(output);
  Accumulate(stats.pipeline.residency.epoch_count, lease.epochs);
  result.status = Status::success();
  result.output_hash = wait.hash.Finish();
  return result;
}

} // namespace rund::compute::detail
