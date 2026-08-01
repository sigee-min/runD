#include "../local.hpp"

#include "../../stats.hpp"
#include "../claim.hpp"
#include "result.hpp"

#include <rund/counter.hpp>

#include <chrono>
#include <limits>
#include <mutex>

namespace rund::compute::detail {

Status start_pipeline(PipelineState &state) noexcept {
  if (state.phase == PipelinePhase::Running) {
    return Status::fail(Reason::PipelineBusy);
  }
  if (state.publication == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  {
    std::unique_lock publication_lock{state.publication->gate,
                                      std::try_to_lock};
    if (!publication_lock.owns_lock() || state.publication->attempt_active) {
      return Status::fail(Reason::PipelineBusy);
    }
    if (state.publication->device_lost) {
      return Status::fail(Reason::DeviceLost);
    }
    if (state.phase == PipelinePhase::Poisoned) {
      return Status::fail(Reason::PipelinePoisoned);
    }
    if (state.publication->generation >= PipelineGenerationCapacity) {
      return Status::fail(Reason::PipelineCapacity);
    }
    if (state.publication->payload_epoch ==
        std::numeric_limits<std::uint64_t>::max()) {
      return Status::fail(Reason::PipelineCapacity);
    }
    if (state.native_generation != state.publication->generation ||
        state.native_parity != state.publication->parity) {
      const Status seeded = seed_pipeline_generations(
          state, state.publication->generation, state.publication->parity);
      if (!seeded) {
        if (seeded.reason() == Reason::DeviceLost) {
          state.publication->device_lost = true;
          state.failure = Reason::DeviceLost;
          state.phase = PipelinePhase::Poisoned;
        }
        return seeded;
      }
    }
    state.attempt_generation = state.publication->generation;
    state.attempt_parity = state.publication->parity;
    state.publication->attempt_active = true;
  }
  reset_pipeline_profile(state);
  reset_pipeline_stats(state);
  const auto claim_begin = std::chrono::steady_clock::now();
  const Status claimed = acquire_pipeline_claims(state);
  const auto claim_end = std::chrono::steady_clock::now();
  state.stats.pipeline.claim_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(claim_end -
                                                           claim_begin)
          .count());
  if (!claimed) {
    {
      std::lock_guard publication_lock{state.publication->gate};
      state.publication->attempt_active = false;
    }
    if (claimed.reason() == Reason::BufferBusy) {
      ::rund::detail::counter::Accumulate(
          state.stats.pipeline.claim_conflict_count, 1u);
    }
    return claimed;
  }
  state.stats.pipeline.verified_step_count = 0u;
  state.stats.pipeline.failed_step_index = PipelineStats::no_failed_step;
  state.verified = 0u;
  state.failure_step_known = false;
  state.writes_possible = false;
  state.backend_submitted = false;
  state.failure = Reason::Ok;
  state.phase = PipelinePhase::Running;
  return Status::success();
}

} // namespace rund::compute::detail
