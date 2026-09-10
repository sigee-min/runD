#include "step/local.hpp"

#include <mutex>

namespace rund::compute::detail {

Status begin_pipeline_step(const std::shared_ptr<PipelineState> &state,
                           const std::size_t index) noexcept {
  if (!valid_pipeline(state) || index >= state->steps.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::lock_guard lock{state->gate};
  if (state->phase != PipelinePhase::Running) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const PipelineStep &step = state->steps[index];
  if (step.route == PipelineRoute::Ordinary) {
    if (index != state->attempt.verified) {
      return Status::fail(Reason::PipelineInvalid);
    }
  } else {
    const PipelineWindow *const descriptor =
        state->windows.find(step.window);
    if (descriptor == nullptr ||
        !async_step_detail::valid_nested_window(state->steps.size(), *descriptor) ||
        state->attempt.verified != descriptor->nested_shape.first()) {
      return Status::fail(Reason::PipelineInvalid);
    }
  }
  begin_pipeline_profile_step(*state, index);
  state->attempt.writes_possible =
      state->attempt.writes_possible ||
      (step.program != nullptr && !step.program->empty() && step.writes);
  return Status::success();
}

} // namespace rund::compute::detail
