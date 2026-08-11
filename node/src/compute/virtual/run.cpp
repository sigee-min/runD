#include "backing.hpp"
#include "local.hpp"
#include "run/backing.hpp"
#include "run/cache.hpp"
#include "run/epoch.hpp"
#include "run/evidence.hpp"
#include "run/projection.hpp"
#include "run/reduce.hpp"
#include "run/scan.hpp"

#include "../../hash/fnv.hpp"
#include "../device/residency_pool.hpp"

#include <mutex>

namespace rund::compute::detail {

Status run_virtual_pipeline(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept {
  return run_virtual_pipeline(state, state == nullptr || state->input == nullptr
                                         ? 0u
                                         : state->input->count);
}

Status run_virtual_pipeline(const std::shared_ptr<VirtualPipelineState> &state,
                            const std::uint64_t active_count) noexcept {
  if (!valid_virtual_pipeline(state)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::unique_lock state_lock{state->gate, std::try_to_lock};
  if (!state_lock.owns_lock() ||
      state->phase == VirtualPipelinePhase::Running) {
    return Status::fail(Reason::PipelineBusy);
  }
  if (state->phase == VirtualPipelinePhase::Poisoned) {
    return Status::fail(Reason::PipelinePoisoned);
  }
  if (active_count > state->input->count ||
      (state->geometry.route != VirtualRoute::Reduction &&
       active_count > state->output->count)) {
    return Status::fail(Reason::ShapeMismatch);
  }
  state->phase = VirtualPipelinePhase::Running;

  Stats stats = begin_virtual_run_evidence(*state, active_count);
  std::uint64_t failed_page = ResidencyStats::no_failed_page;
  bool poison_pipeline = false;
  const auto finish = [&](const Status status,
                          const std::uint64_t output_hash = 0u) noexcept {
    return publish_virtual_run_evidence(*state, stats, status, failed_page,
                                        output_hash, poison_pipeline);
  };

  VirtualBacking &input_backing = *state->input->backing;
  VirtualBacking &output_backing = *state->output->backing;
  std::scoped_lock backing_locks{VirtualBackingAccess::gate(input_backing),
                                 VirtualBackingAccess::gate(output_backing)};

  if (state->pipeline->residency_pool == nullptr) {
    poison_pipeline = true;
    return finish(Status::fail(Reason::PipelineInvalid));
  }
  std::unique_lock pool_lock{state->pipeline->residency_pool->execution_gate,
                             std::try_to_lock};
  if (!pool_lock.owns_lock()) {
    return finish(Status::fail(Reason::PipelineBusy));
  }

  VirtualRunProjection run{};
  if (!project_virtual_run(*state, active_count, run)) {
    poison_pipeline = true;
    return finish(Status::fail(Reason::PipelineInvalid));
  }
  stats.pipeline.residency.resident_frames_peak =
      run.active.resident_frames_peak;
  VirtualReduction reduction{};
  if (run.reduction) {
    const Status initialized = begin_virtual_reduction(run, reduction);
    if (!initialized) {
      return finish(initialized);
    }
  }
  VirtualScan scan{};
  if (run.scan) {
    const Status initialized = begin_virtual_scan(run, scan);
    if (!initialized) {
      return finish(initialized);
    }
  }
  const Status recovery = validate_virtual_recovery(
      input_backing, output_backing, run.active.output_bytes);
  if (!recovery) {
    return finish(recovery);
  }

  ::rund::node::hash_detail::Fnv output_hash{};
  if (active_count == 0u) {
    if (run.reduction) {
      const Status reduced =
          finish_virtual_reduction(output_backing, run, reduction,
                                   stats.pipeline.residency, output_hash);
      if (!reduced) {
        return finish(reduced);
      }
      clear_virtual_recovery(output_backing);
    }
    return finish(Status::success(), output_hash.Finish());
  }
  if (!bind_virtual_run_transfer(*state, run)) {
    poison_pipeline = true;
    return finish(Status::fail(Reason::PipelineInvalid));
  }
  std::array<bool, 2u> prefetch_pending{};
  for (std::uint64_t epoch = 0u; epoch < run.active.stream.epoch_count();
       ++epoch) {
    const VirtualEpochResult result = execute_virtual_epoch(
        *state, input_backing, output_backing, run, epoch, prefetch_pending,
        stats, output_hash, run.reduction ? &reduction : nullptr,
        run.scan ? &scan : nullptr);
    if (!result.status) {
      failed_page = result.failed_page;
      poison_pipeline = result.poison_pipeline;
      return finish(result.status);
    }
  }

  if (run.reduction) {
    const Status reduced = finish_virtual_reduction(
        output_backing, run, reduction, stats.pipeline.residency, output_hash);
    if (!reduced) {
      failed_page = run.active.stream.page_count() == 0u
                        ? ResidencyStats::no_failed_page
                        : run.active.stream.page_count() - 1u;
      return finish(reduced);
    }
  } else {
    const residency::AuthorityResult drain =
        state->pipeline->residency_pool->authority.drain_dirty();
    if (!drain) {
      poison_pipeline = true;
      return finish(Status::fail(Reason::PipelineInvalid));
    }
    const Status written = writeback_residency_cache(
        output_backing, run, drain.lease.transitions, stats.pipeline.residency);
    const bool completed =
        written ? state->pipeline->residency_pool->authority.complete(
                      drain.lease.token, true)
                : state->pipeline->residency_pool->authority.discard(
                      drain.lease.token);
    if (!written || !completed) {
      failed_page = drain.lease.transitions.empty()
                        ? ResidencyStats::no_failed_page
                        : drain.lease.transitions.front().key.page;
      poison_pipeline = !completed;
      return finish(written ? Status::fail(Reason::PipelineInvalid) : written);
    }
  }

  clear_virtual_recovery(output_backing);
  return finish(Status::success(), output_hash.Finish());
}

} // namespace rund::compute::detail
