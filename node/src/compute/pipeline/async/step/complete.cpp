#include "local.hpp"

namespace rund::compute::detail::async_step_detail {

Status
complete_pipeline_step_locked(PipelineState &state, const std::size_t index,
                              const Status result,
                              CpuPipelineSchedule *const schedule) noexcept {
  if (state.phase != PipelinePhase::Running || index >= state.steps.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  PipelineStep &step = state.steps[index];
  const PipelineWindow *descriptor = nullptr;
  std::size_t expected_fold = 0u;
  std::uint32_t expected_fold_route = 0u;
  const bool nested = step.route != PipelineRoute::Ordinary;
  if (!nested) {
    if (index != state.attempt.verified ||
        (schedule != nullptr && schedule->step != index)) {
      return Status::fail(Reason::PipelineInvalid);
    }
  } else {
    descriptor = state.windows.find(step.window);
    if (schedule == nullptr || schedule->step != index ||
        descriptor == nullptr ||
        !valid_nested_window(state.steps.size(), *descriptor) ||
        state.attempt.verified != descriptor->nested_shape.first() ||
        schedule->outer >= descriptor->nested_shape.outer_bound()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    if (!descriptor->nested_shape.fold_route_for_outer(
            static_cast<std::uint32_t>(schedule->outer), expected_fold_route)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    expected_fold = descriptor->nested_shape.fold_first() + expected_fold_route;
    if ((step.route == PipelineRoute::NestedSeed &&
         index != descriptor->nested_shape.seed_first() + schedule->outer) ||
        (step.route == PipelineRoute::NestedAction &&
         (index < descriptor->nested_shape.action_first() ||
          index >= descriptor->nested_shape.fold_first())) ||
        (step.route == PipelineRoute::NestedFold && index != expected_fold)) {
      return Status::fail(Reason::PipelineInvalid);
    }
  }

  state.attempt.writes_possible = state.attempt.writes_possible || step.writes;
  Reason semantic = result.reason();
  if (result) {
    const std::shared_ptr<JobState> job = selected_pipeline_job(state, step);
    const Status published = publish_cpu_pipeline_views(job);
    if (published) {
      const Status consumed = consume_cpu_pipeline_step(state, index);
      semantic = consumed.reason();
    } else {
      semantic = published.reason();
    }
  }
  if (semantic == Reason::Ok && descriptor != nullptr &&
      step.route == PipelineRoute::NestedFold) {
    bool window_wrote = false;
    const Status published = publish_cpu_pipeline_window(
        state, step.window, schedule->outer, window_wrote);
    state.attempt.writes_possible =
        state.attempt.writes_possible || window_wrote;
    semantic = published.reason();
  }
  const Status outcome =
      semantic == Reason::Ok ? Status::success() : Status::fail(semantic);
  capture_cpu_pipeline_step(state, index, true);
  finish_pipeline_profile_step(state, index);
  if (!outcome) {
    const bool inner_known =
        descriptor != nullptr && step.route == PipelineRoute::NestedAction;
    const std::size_t inner =
        inner_known ? index - descriptor->nested_shape.action_first() : 0u;
    record_pipeline_failure(state, index, descriptor != nullptr,
                            descriptor == nullptr ? 0u : schedule->outer,
                            inner_known, inner);
    if (descriptor != nullptr) {
      state.windows.progress(step.window - 1u).stopped = true;
    }
    return outcome;
  }

  if (!nested) {
    ++state.attempt.verified;
    state.stats.pipeline.verified_step_count =
        logical_verified_steps(state, state.attempt.verified);
    if (schedule != nullptr) {
      ++schedule->step;
    }
    return Status::success();
  }

  switch (step.route) {
  case PipelineRoute::NestedSeed:
    schedule->step = descriptor->nested_shape.action_first();
    break;
  case PipelineRoute::NestedAction:
    ++state.stats.pipeline.executed_inner_iteration_count;
    schedule->step = index + 1u < descriptor->nested_shape.fold_first()
                         ? index + 1u
                         : expected_fold;
    break;
  case PipelineRoute::NestedFold:
    state.windows.progress(step.window - 1u).current =
        expected_fold_route == 1u ? PipelineWindow::second
                                  : PipelineWindow::first;
    ++state.stats.pipeline.executed_outer_window_count;
    ::rund::detail::counter::Accumulate(state.stats.control.iteration_count,
                                        1u);
    ++schedule->outer;
    schedule->step = descriptor->nested_shape.seed_first();
    break;
  case PipelineRoute::Ordinary:
    return Status::fail(Reason::PipelineInvalid);
  }
  return Status::success();
}

} // namespace rund::compute::detail::async_step_detail
