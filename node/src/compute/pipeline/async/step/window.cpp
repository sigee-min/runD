#include "local.hpp"

namespace rund::compute::detail::async_step_detail {

bool valid_nested_window(const std::size_t step_count,
                         const PipelineWindow &window) noexcept {
  node::accel::detail::NestedTemplateShape expected{};
  return window.nested() && window.recurrent_output_count != 0u &&
         window.nested_shape.end() <= step_count &&
         node::accel::detail::ProveNestedTemplateShape(
             window.nested_shape.first(), window.control.maximum,
             window.control.tile, window.nested_shape.inner_bound(),
             expected) &&
         expected == window.nested_shape;
}

std::shared_ptr<JobState>
selected_pipeline_job(PipelineState &state, const PipelineStep &step) noexcept {
  return state.transactional && state.attempt.parity != 0u ? step.alternate_job
                                                           : step.job;
}

} // namespace rund::compute::detail::async_step_detail
