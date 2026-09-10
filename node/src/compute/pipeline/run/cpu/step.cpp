#include "internal.hpp"

#include "../../state.hpp"

#include "../../../job/local.hpp"

#include <cstddef>
#include <memory>

namespace rund::compute::detail {

Status execute_cpu_pipeline_step(PipelineState &state, PipelineOutcome &outcome,
                                 const std::size_t index) {
  if (index >= state.steps.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  PipelineStep &step = state.steps[index];
  begin_pipeline_profile_step(state, index);
  const auto finish_step = [&](const bool executed) noexcept {
    capture_cpu_pipeline_step(state, index, executed);
    finish_pipeline_profile_step(state, index);
  };
  outcome.writes_possible = outcome.writes_possible || step.writes;
  const std::shared_ptr<JobState> &job =
      state.transactional && state.attempt.parity != 0u ? step.alternate_job
                                                        : step.job;
  const Status gathered = gather_cpu_pipeline_views(job);
  if (!gathered) {
    finish_step(false);
    return gathered;
  }
  const Status ran = run_pipeline_job(job);
  if (!ran) {
    const Status consumed = consume_cpu_pipeline_step(state, index, ran);
    finish_step(true);
    return consumed ? ran : consumed;
  }
  const Status published = publish_cpu_pipeline_views(job);
  if (!published) {
    finish_step(true);
    return published;
  }
  const Status consumed = consume_cpu_pipeline_step(state, index);
  finish_step(true);
  return consumed;
}

} // namespace rund::compute::detail
