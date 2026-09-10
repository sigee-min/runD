#include "local.hpp"

namespace rund::compute::detail::async_step_detail {

void record_pipeline_failure(PipelineState &state, const std::size_t index,
                             const bool outer_known, const std::size_t outer,
                             const bool inner_known,
                             const std::size_t inner) noexcept {
  state.attempt.failure_step_known = index < state.steps.size();
  state.stats.pipeline.verified_step_count =
      logical_verified_steps(state, state.attempt.verified);
  state.stats.pipeline.failed_step_index = logical_step_index(state, index);
  if (index < state.steps.size()) {
    state.stats.pipeline.failed_nested_phase =
        pipeline_nested_phase(state.steps[index].route);
  }
  if (outer_known) {
    state.stats.pipeline.failed_outer_window = outer;
  }
  if (inner_known) {
    state.stats.pipeline.failed_inner_iteration = inner;
  }
}

} // namespace rund::compute::detail::async_step_detail
