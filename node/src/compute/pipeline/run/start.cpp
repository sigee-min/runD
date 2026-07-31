#include "../local.hpp"

#include "../../stats.hpp"
#include "../claim.hpp"
#include "result.hpp"

#include <rund/counter.hpp>

#include <chrono>

namespace rund::compute::detail {

Status start_pipeline(PipelineState &state) noexcept {
  if (state.phase == PipelinePhase::Running) {
    return Status::fail(Reason::PipelineBusy);
  }
  if (state.phase == PipelinePhase::Poisoned) {
    return Status::fail(Reason::PipelinePoisoned);
  }
  reset_pipeline_profile(state);
  if (state.device_lost) {
    return Status::fail(Reason::DeviceLost);
  }
  reset_pipeline_stats(state);
  const auto claim_begin = std::chrono::steady_clock::now();
  const Status claimed = acquire_pipeline_claims(state);
  const auto claim_end = std::chrono::steady_clock::now();
  state.stats.pipeline.claim_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(claim_end -
                                                           claim_begin)
          .count());
  if (!claimed) {
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
