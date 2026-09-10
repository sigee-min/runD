#include "local.hpp"

#include <mutex>

namespace rund::compute::detail {

Status
initialize_cpu_pipeline_schedule(const std::shared_ptr<PipelineState> &state,
                                 CpuPipelineSchedule &schedule) noexcept {
  if (!valid_pipeline(state)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::lock_guard lock{state->gate};
  if (state->phase != PipelinePhase::Running ||
      state->device->backend != Backend::Cpu || state->attempt.verified != 0u) {
    return Status::fail(Reason::PipelineInvalid);
  }
  schedule = {};
  return Status::success();
}

CpuPipelineSelection
select_cpu_pipeline_step(const std::shared_ptr<PipelineState> &state,
                         CpuPipelineSchedule &schedule) noexcept {
  using namespace async_step_detail;
  if (!valid_pipeline(state)) {
    return CpuPipelineSelection::failed(Status::fail(Reason::PipelineInvalid));
  }
  std::lock_guard lock{state->gate};
  if (state->phase != PipelinePhase::Running ||
      state->device->backend != Backend::Cpu) {
    return CpuPipelineSelection::failed(Status::fail(Reason::PipelineInvalid));
  }

  for (;;) {
    if (schedule.step == state->steps.size()) {
      return state->attempt.verified == state->steps.size()
                 ? CpuPipelineSelection::complete()
                 : CpuPipelineSelection::failed(
                       Status::fail(Reason::PipelineInvalid));
    }
    if (schedule.step > state->steps.size()) {
      return CpuPipelineSelection::failed(
          Status::fail(Reason::PipelineInvalid));
    }

    PipelineStep &step = state->steps[schedule.step];
    if (step.route == PipelineRoute::Ordinary) {
      if (schedule.outer != 0u || schedule.step != state->attempt.verified) {
        record_pipeline_failure(*state, schedule.step);
        return CpuPipelineSelection::failed(
            Status::fail(Reason::PipelineInvalid));
      }
      bool active = true;
      const Status ready =
          prepare_cpu_pipeline_window(*state, schedule.step, active);
      if (!ready) {
        record_pipeline_failure(*state, schedule.step);
        return CpuPipelineSelection::failed(
            pipeline_window_status(state->windows, step, ready, state->stats.control));
      }
      if (!active) {
        ++state->attempt.verified;
        state->stats.pipeline.verified_step_count =
            logical_verified_steps(*state, state->attempt.verified);
        ++schedule.step;
        continue;
      }
      const std::shared_ptr<JobState> job = selected_pipeline_job(*state, step);
      if (job == nullptr) {
        record_pipeline_failure(*state, schedule.step);
        return CpuPipelineSelection::failed(
            Status::fail(Reason::PipelineInvalid));
      }
      return CpuPipelineSelection::selected(job, schedule.step);
    }

    if (step.route != PipelineRoute::NestedSeed) {
      const PipelineWindow *const descriptor = state->windows.find(step.window);
      if (descriptor == nullptr ||
          !valid_nested_window(state->steps.size(), *descriptor) ||
          state->attempt.verified != descriptor->nested_shape.first() ||
          schedule.outer >= descriptor->nested_shape.outer_bound()) {
        record_pipeline_failure(*state, schedule.step);
        return CpuPipelineSelection::failed(
            Status::fail(Reason::PipelineInvalid));
      }
      const std::shared_ptr<JobState> job = selected_pipeline_job(*state, step);
      if (job == nullptr) {
        record_pipeline_failure(*state, schedule.step, true, schedule.outer);
        return CpuPipelineSelection::failed(
            Status::fail(Reason::PipelineInvalid));
      }
      return CpuPipelineSelection::selected(job, schedule.step);
    }

    const PipelineWindow *const descriptor = state->windows.find(step.window);
    if (descriptor == nullptr ||
        !valid_nested_window(state->steps.size(), *descriptor) ||
        state->attempt.verified != descriptor->nested_shape.first() ||
        schedule.outer > descriptor->nested_shape.outer_bound() ||
        (schedule.step != descriptor->nested_shape.seed_first() &&
         (schedule.outer >= descriptor->nested_shape.outer_bound() ||
          schedule.step !=
              descriptor->nested_shape.seed_first() + schedule.outer))) {
      record_pipeline_failure(*state, schedule.step, true, schedule.outer);
      return CpuPipelineSelection::failed(
          Status::fail(Reason::PipelineInvalid));
    }
    if (schedule.outer == descriptor->nested_shape.outer_bound()) {
      state->attempt.verified = descriptor->nested_shape.end();
      state->stats.pipeline.verified_step_count =
          logical_verified_steps(*state, state->attempt.verified);
      schedule.step = descriptor->nested_shape.end();
      schedule.outer = 0u;
      continue;
    }

    schedule.step = descriptor->nested_shape.seed_first() + schedule.outer;
    const PipelineStep &occurrence =
        state->steps[descriptor->nested_shape.fold_first()];
    bool active = true;
    const Status ready = prepare_cpu_pipeline_window(
        cpu_pipeline_publication_context(*state), state->windows,
        occurrence.window, static_cast<std::uint32_t>(schedule.outer),
        state->stats.control, active);
    if (!ready) {
      record_pipeline_failure(*state, schedule.step, true, schedule.outer);
      state->windows.progress(step.window - 1u).stopped = true;
      return CpuPipelineSelection::failed(pipeline_window_status(
          state->windows, occurrence, ready, state->stats.control));
    }
    if (!active) {
      ::rund::detail::counter::Accumulate(
          state->stats.pipeline.skipped_outer_window_count, 1u);
      ::rund::detail::counter::Accumulate(
          state->stats.pipeline.skipped_inner_iteration_count,
          descriptor->nested_shape.action_count());
      ++schedule.outer;
      schedule.step = descriptor->nested_shape.seed_first();
      continue;
    }
    PipelineStep &selected = state->steps[schedule.step];
    if (selected.route != PipelineRoute::NestedSeed) {
      record_pipeline_failure(*state, schedule.step, true, schedule.outer);
      return CpuPipelineSelection::failed(
          Status::fail(Reason::PipelineInvalid));
    }
    const std::shared_ptr<JobState> job =
        selected_pipeline_job(*state, selected);
    if (job == nullptr) {
      record_pipeline_failure(*state, schedule.step, true, schedule.outer);
      return CpuPipelineSelection::failed(
          Status::fail(Reason::PipelineInvalid));
    }
    return CpuPipelineSelection::selected(job, schedule.step);
  }
}

Status
complete_cpu_pipeline_schedule_step(const std::shared_ptr<PipelineState> &state,
                                    CpuPipelineSchedule &schedule,
                                    const Status result) noexcept {
  if (!valid_pipeline(state) || schedule.step >= state->steps.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::lock_guard lock{state->gate};
  return async_step_detail::complete_pipeline_step_locked(*state, schedule.step,
                                                          result, &schedule);
}

} // namespace rund::compute::detail
