#include "../local.hpp"

#include "../../stats.hpp"
#include "../claim.hpp"
#include "result.hpp"

#include <rund/counter.hpp>

#include <chrono>
#include <limits>
#include <mutex>

namespace rund::compute::detail {

Status start_pipeline(PipelineState &state,
                      const PipelineClaimAuthority authority,
                      const std::uint64_t control_generation,
                      const std::uint8_t control_parity) noexcept {
  if (state.phase == PipelinePhase::Running) {
    return Status::fail(Reason::PipelineBusy);
  }
  if (state.publication == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const bool explicit_control = control_generation != PipelineNoGeneration;
  if (explicit_control && control_parity > 1u) {
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
    if (state.publication->generation >= PipelineGenerationCapacity ||
        (explicit_control &&
         control_generation >= PipelineGenerationCapacity)) {
      return Status::fail(Reason::PipelineCapacity);
    }
    if (state.publication->payload_epoch ==
        std::numeric_limits<std::uint64_t>::max()) {
      return Status::fail(Reason::PipelineCapacity);
    }
    const std::uint64_t selected_generation =
        explicit_control ? control_generation : state.publication->generation;
    const std::uint8_t selected_parity =
        explicit_control ? control_parity : state.publication->parity;
    if (state.native_generation != selected_generation ||
        state.native_parity != selected_parity) {
      const Status seeded = seed_pipeline_generations(
          state, selected_generation, selected_parity);
      if (!seeded) {
        if (seeded.reason() == Reason::DeviceLost) {
          state.publication->device_lost = true;
          state.failure = Reason::DeviceLost;
          state.phase = PipelinePhase::Poisoned;
        }
        return seeded;
      }
    }
    state.attempt.generation = selected_generation;
    state.attempt.parity = selected_parity;
    state.publication->attempt_active = true;
  }
  reset_pipeline_profile(state);
  reset_pipeline_stats(state);
  Status claimed = Status::success();
  if (authority == PipelineClaimAuthority::PrivateResidency) {
    claimed = has_private_residency_authority(state)
                  ? Status::success()
                  : Status::fail(Reason::PipelineInvalid);
    state.stats.pipeline.claim_ns = 0u;
  } else {
    const auto claim_begin = std::chrono::steady_clock::now();
    claimed = acquire_pipeline_claims(state);
    const auto claim_end = std::chrono::steady_clock::now();
    state.stats.pipeline.claim_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(claim_end -
                                                             claim_begin)
            .count());
  }
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
  state.attempt = PipelineAttemptState{.generation = state.attempt.generation,
                                       .parity = state.attempt.parity};
  if (state.device->backend == Backend::Cpu) {
    state.windows.reset_progress();
  }
  state.failure = Reason::Ok;
  state.phase = PipelinePhase::Running;
  return Status::success();
}

} // namespace rund::compute::detail
