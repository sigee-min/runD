#include "local.hpp"

#include <rund/compute/pipeline.hpp>
#include <rund/counter.hpp>

#include "../backend.hpp"
#include "../job/local.hpp"
#include "../stats.hpp"
#include "../status.hpp"
#include "claim.hpp"
#include "run/result.hpp"

#include <algorithm>
#include <mutex>
#include <utility>

namespace rund::compute::detail {
namespace {

[[nodiscard]] PipelineWindow *
pipeline_window(PipelineState &state, const PipelineStep &step) noexcept {
  return step.window == 0u || step.window > state.windows.size()
             ? nullptr
             : &state.windows[step.window - 1u];
}

[[nodiscard]] bool valid_nested_window(const PipelineState &state,
                                       const PipelineWindow &window) noexcept {
  return window.nested && window.seed_count != 0u &&
         window.begin == window.seed_first &&
         window.action_first == window.seed_first + window.seed_count &&
         window.fold_first == window.action_first + window.action_count &&
         window.end == window.fold_first + 3u &&
         window.end <= state.steps.size();
}

[[nodiscard]] std::shared_ptr<JobState>
selected_pipeline_job(PipelineState &state, const PipelineStep &step) noexcept {
  return state.transactional && state.attempt_parity != 0u ? step.alternate_job
                                                           : step.job;
}

void record_pipeline_failure(PipelineState &state, const std::size_t index,
                             const bool outer_known = false,
                             const std::size_t outer = 0u,
                             const bool inner_known = false,
                             const std::size_t inner = 0u) noexcept {
  state.failure_step_known = index < state.steps.size();
  state.stats.pipeline.verified_step_count =
      logical_verified_steps(state, state.verified);
  state.stats.pipeline.failed_step_index = logical_step_index(state, index);
  if (index < state.steps.size()) {
    switch (state.steps[index].route) {
    case PipelineRoute::NestedSeed:
      state.stats.pipeline.failed_nested_phase = PipelineNestedPhase::Seed;
      break;
    case PipelineRoute::NestedAction:
      state.stats.pipeline.failed_nested_phase = PipelineNestedPhase::Action;
      break;
    case PipelineRoute::NestedFold:
      state.stats.pipeline.failed_nested_phase = PipelineNestedPhase::Fold;
      break;
    case PipelineRoute::Ordinary:
      state.stats.pipeline.failed_nested_phase = PipelineNestedPhase::None;
      break;
    }
  }
  if (outer_known) {
    state.stats.pipeline.failed_outer_window = outer;
  }
  if (inner_known) {
    state.stats.pipeline.failed_inner_iteration = inner;
  }
}

[[nodiscard]] Status
complete_pipeline_step_locked(PipelineState &state, const std::size_t index,
                              const Status result,
                              CpuPipelineSchedule *const schedule) noexcept {
  if (state.phase != PipelinePhase::Running || index >= state.steps.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  PipelineStep &step = state.steps[index];
  PipelineWindow *descriptor = nullptr;
  const bool nested = step.route != PipelineRoute::Ordinary;
  if (!nested) {
    if (index != state.verified ||
        (schedule != nullptr && schedule->step != index)) {
      return Status::fail(Reason::PipelineInvalid);
    }
  } else {
    descriptor = pipeline_window(state, step);
    if (schedule == nullptr || schedule->step != index ||
        descriptor == nullptr || !valid_nested_window(state, *descriptor) ||
        state.verified != descriptor->begin ||
        schedule->outer >= descriptor->seed_count) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const std::size_t expected_fold =
        descriptor->fold_first +
        (schedule->outer == 0u ? 0u : ((schedule->outer & 1u) != 0u ? 1u : 2u));
    if ((step.route == PipelineRoute::NestedSeed &&
         index != descriptor->seed_first + schedule->outer) ||
        (step.route == PipelineRoute::NestedAction &&
         (index < descriptor->action_first ||
          index >= descriptor->fold_first)) ||
        (step.route == PipelineRoute::NestedFold && index != expected_fold)) {
      return Status::fail(Reason::PipelineInvalid);
    }
  }

  state.writes_possible = state.writes_possible || step.writes;
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
    state.writes_possible = state.writes_possible || window_wrote;
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
        inner_known ? index - descriptor->action_first : 0u;
    record_pipeline_failure(state, index, descriptor != nullptr,
                            descriptor == nullptr ? 0u : schedule->outer,
                            inner_known, inner);
    if (descriptor != nullptr) {
      descriptor->stopped = true;
    }
    return outcome;
  }

  if (!nested) {
    ++state.verified;
    state.stats.pipeline.verified_step_count =
        logical_verified_steps(state, state.verified);
    if (schedule != nullptr) {
      ++schedule->step;
    }
    return Status::success();
  }

  switch (step.route) {
  case PipelineRoute::NestedSeed:
    schedule->step = descriptor->action_first;
    break;
  case PipelineRoute::NestedAction:
    ++state.stats.pipeline.executed_inner_iteration_count;
    schedule->step = index + 1u < descriptor->fold_first
                         ? index + 1u
                         : descriptor->fold_first +
                               (schedule->outer == 0u
                                    ? 0u
                                    : ((schedule->outer & 1u) != 0u ? 1u : 2u));
    break;
  case PipelineRoute::NestedFold:
    descriptor->current = PipelineWindow::first +
                          static_cast<std::uint32_t>(schedule->outer & 1u);
    ++state.stats.pipeline.executed_outer_window_count;
    ::rund::detail::counter::Accumulate(state.stats.control.iteration_count,
                                        1u);
    ++schedule->outer;
    schedule->step = descriptor->seed_first;
    break;
  case PipelineRoute::Ordinary:
    return Status::fail(Reason::PipelineInvalid);
  }
  return Status::success();
}

} // namespace

Result<Backend>
pipeline_backend(const std::shared_ptr<PipelineState> &state) noexcept {
  return !valid_pipeline(state)
             ? Result<Backend>::fail(Reason::PipelineInvalid)
             : Result<Backend>::success(state->device->backend);
}

kernel::u32
pipeline_workers(const std::shared_ptr<PipelineState> &state) noexcept {
  if (!valid_pipeline(state)) {
    return 0u;
  }
  const CpuDeviceState *const cpu = cpu_device(*state->device);
  return cpu == nullptr ? 0u : cpu->workers.requested_worker_width;
}

Status queue_pipeline(const std::shared_ptr<PipelineState> &state) noexcept {
  if (!valid_pipeline(state)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::lock_guard lock{state->gate};
  return start_pipeline(*state);
}

Status cancel_pipeline(const std::shared_ptr<PipelineState> &state) noexcept {
  if (!valid_pipeline(state)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::lock_guard lock{state->gate};
  if (state->phase != PipelinePhase::Running) {
    return state->failure == Reason::Cancelled ? Status::fail(Reason::Cancelled)
           : state->phase == PipelinePhase::Poisoned
               ? Status::fail(Reason::PipelinePoisoned)
               : Status::fail(Reason::AlreadyCompleted);
  }
  publish_pipeline_terminal(
      *state,
      PipelineTerminal{
          .reason = Reason::Cancelled,
          .verified = state->verified,
          .failed_step = state->verified,
          .failure_step_known = state->failure_step_known,
          .writes_possible = state->writes_possible || state->backend_submitted,
          .publication_suppressed = state->device->backend == Backend::Cpu ||
                                    !state->backend_submitted});
  return Status::fail(Reason::Cancelled);
}

Status fail_pipeline(const std::shared_ptr<PipelineState> &state,
                     const Status failure) noexcept {
  if (!valid_pipeline(state)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::lock_guard lock{state->gate};
  if (state->phase == PipelinePhase::Running) {
    publish_pipeline_terminal(
        *state,
        PipelineTerminal{.reason = failure.reason(),
                         .verified = state->verified,
                         .failed_step = state->verified,
                         .failure_step_known = state->failure_step_known,
                         .writes_possible =
                             state->writes_possible || state->backend_submitted,
                         .publication_suppressed =
                             state->device->backend == Backend::Cpu ||
                             !state->backend_submitted});
  }
  return Status::fail(failure.reason());
}

std::size_t
pipeline_size(const std::shared_ptr<PipelineState> &state) noexcept {
  return state == nullptr ? 0u : state->steps.size();
}

std::shared_ptr<JobState>
pipeline_job(const std::shared_ptr<PipelineState> &state,
             const std::size_t index) noexcept {
  return state == nullptr || index >= state->steps.size()
             ? std::shared_ptr<JobState>{}
         : state->transactional && state->attempt_parity != 0u
             ? state->steps[index].alternate_job
             : state->steps[index].job;
}

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
    if (index != state->verified) {
      return Status::fail(Reason::PipelineInvalid);
    }
  } else {
    const PipelineWindow *const descriptor = pipeline_window(*state, step);
    if (descriptor == nullptr || !valid_nested_window(*state, *descriptor) ||
        state->verified != descriptor->begin) {
      return Status::fail(Reason::PipelineInvalid);
    }
  }
  begin_pipeline_profile_step(*state, index);
  state->writes_possible =
      state->writes_possible ||
      (step.program != nullptr && !step.program->empty() && step.writes);
  return Status::success();
}

Status complete_pipeline_step(const std::shared_ptr<PipelineState> &state,
                              const std::size_t index,
                              const Status result) noexcept {
  if (!valid_pipeline(state) || index >= state->steps.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::lock_guard lock{state->gate};
  return complete_pipeline_step_locked(*state, index, result, nullptr);
}

Status
initialize_cpu_pipeline_schedule(const std::shared_ptr<PipelineState> &state,
                                 CpuPipelineSchedule &schedule) noexcept {
  if (!valid_pipeline(state)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::lock_guard lock{state->gate};
  if (state->phase != PipelinePhase::Running ||
      state->device->backend != Backend::Cpu || state->verified != 0u) {
    return Status::fail(Reason::PipelineInvalid);
  }
  schedule = {};
  reset_cpu_resident(*state);
  return Status::success();
}

CpuPipelineSelection
select_cpu_pipeline_step(const std::shared_ptr<PipelineState> &state,
                         CpuPipelineSchedule &schedule) noexcept {
  if (!valid_pipeline(state)) {
    return {.status = Status::fail(Reason::PipelineInvalid)};
  }
  std::lock_guard lock{state->gate};
  if (state->phase != PipelinePhase::Running ||
      state->device->backend != Backend::Cpu) {
    return {.status = Status::fail(Reason::PipelineInvalid)};
  }

  for (;;) {
    if (schedule.step == state->steps.size()) {
      return state->verified == state->steps.size()
                 ? CpuPipelineSelection{.step = schedule.step, .complete = true}
                 : CpuPipelineSelection{
                       .status = Status::fail(Reason::PipelineInvalid)};
    }
    if (schedule.step > state->steps.size()) {
      return {.status = Status::fail(Reason::PipelineInvalid)};
    }

    PipelineStep &step = state->steps[schedule.step];
    if (step.route == PipelineRoute::Ordinary) {
      if (schedule.outer != 0u || schedule.step != state->verified) {
        record_pipeline_failure(*state, schedule.step);
        return {.status = Status::fail(Reason::PipelineInvalid)};
      }
      bool active = true;
      const Status ready =
          prepare_cpu_pipeline_window(*state, schedule.step, active);
      if (!ready) {
        record_pipeline_failure(*state, schedule.step);
        return {.status =
                    pipeline_window_status(*state, step, ready, state->stats)};
      }
      if (!active) {
        ++state->verified;
        state->stats.pipeline.verified_step_count =
            logical_verified_steps(*state, state->verified);
        ++schedule.step;
        continue;
      }
      const std::shared_ptr<JobState> job = selected_pipeline_job(*state, step);
      if (job == nullptr) {
        record_pipeline_failure(*state, schedule.step);
        return {.status = Status::fail(Reason::PipelineInvalid)};
      }
      return {.job = job, .step = schedule.step};
    }

    if (step.route != PipelineRoute::NestedSeed) {
      PipelineWindow *const descriptor = pipeline_window(*state, step);
      if (descriptor == nullptr || !valid_nested_window(*state, *descriptor) ||
          state->verified != descriptor->begin ||
          schedule.outer >= descriptor->seed_count) {
        record_pipeline_failure(*state, schedule.step);
        return {.status = Status::fail(Reason::PipelineInvalid)};
      }
      const std::shared_ptr<JobState> job = selected_pipeline_job(*state, step);
      if (job == nullptr) {
        record_pipeline_failure(*state, schedule.step, true, schedule.outer);
        return {.status = Status::fail(Reason::PipelineInvalid)};
      }
      return {.job = job, .step = schedule.step};
    }

    PipelineWindow *const descriptor = pipeline_window(*state, step);
    if (descriptor == nullptr || !valid_nested_window(*state, *descriptor) ||
        state->verified != descriptor->begin ||
        schedule.outer > descriptor->seed_count ||
        (schedule.step != descriptor->seed_first &&
         (schedule.outer >= descriptor->seed_count ||
          schedule.step != descriptor->seed_first + schedule.outer))) {
      record_pipeline_failure(*state, schedule.step, true, schedule.outer);
      return {.status = Status::fail(Reason::PipelineInvalid)};
    }
    if (schedule.outer == descriptor->seed_count) {
      state->verified = descriptor->end;
      state->stats.pipeline.verified_step_count =
          logical_verified_steps(*state, state->verified);
      schedule.step = descriptor->end;
      schedule.outer = 0u;
      continue;
    }

    schedule.step = descriptor->seed_first + schedule.outer;
    PipelineStep occurrence = state->steps[descriptor->fold_first];
    occurrence.iteration = static_cast<std::uint32_t>(schedule.outer);
    occurrence.iteration_bound =
        static_cast<std::uint32_t>(descriptor->seed_count);
    bool active = true;
    const Status ready =
        prepare_cpu_pipeline_window(*state, occurrence, active);
    if (!ready) {
      record_pipeline_failure(*state, schedule.step, true, schedule.outer);
      descriptor->stopped = true;
      return {.status = pipeline_window_status(*state, occurrence, ready,
                                               state->stats)};
    }
    if (!active) {
      ::rund::detail::counter::Accumulate(
          state->stats.pipeline.skipped_outer_window_count, 1u);
      ::rund::detail::counter::Accumulate(
          state->stats.pipeline.skipped_inner_iteration_count,
          static_cast<std::uint64_t>(descriptor->action_count));
      ++schedule.outer;
      schedule.step = descriptor->seed_first;
      continue;
    }
    PipelineStep &selected = state->steps[schedule.step];
    if (selected.route != PipelineRoute::NestedSeed) {
      record_pipeline_failure(*state, schedule.step, true, schedule.outer);
      return {.status = Status::fail(Reason::PipelineInvalid)};
    }
    const std::shared_ptr<JobState> job =
        selected_pipeline_job(*state, selected);
    if (job == nullptr) {
      record_pipeline_failure(*state, schedule.step, true, schedule.outer);
      return {.status = Status::fail(Reason::PipelineInvalid)};
    }
    return {.job = job, .step = schedule.step};
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
  return complete_pipeline_step_locked(*state, schedule.step, result,
                                       &schedule);
}

Status
complete_cpu_pipeline(const std::shared_ptr<PipelineState> &state) noexcept {
  if (!valid_pipeline(state)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::lock_guard lock{state->gate};
  if (state->phase != PipelinePhase::Running ||
      state->verified != state->steps.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  state->stats.command_submits = 0u;
  const Status published = publish_cpu_pipeline(*state);
  if (!published) {
    publish_pipeline_terminal(
        *state, PipelineTerminal{.reason = published.reason(),
                                 .verified = state->verified,
                                 .failed_step = state->steps.empty()
                                                    ? 0u
                                                    : state->steps.size() - 1u,
                                 .failure_step_known = !state->steps.empty(),
                                 .writes_possible = true,
                                 .publication_suppressed = false});
    return published;
  }
  publish_pipeline_terminal(*state, PipelineTerminal{});
  return Status::success();
}

Status submit_pipeline_on(const std::shared_ptr<PipelineState> &state,
                          std::shared_ptr<void> lifetime,
                          const PipelineCompletion completion,
                          void *const user) noexcept {
  if (!valid_pipeline(state) || completion == nullptr || user == nullptr ||
      state->device->backend == Backend::Cpu) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const DeviceOps *const ops = state->device->ops;
  if (ops == nullptr || ops->submit_pipeline == nullptr) {
    return Status::fail(Reason::AccelProgramInvalid);
  }
  const node::accel::detail::PreparedKernelPipeline *prepared = nullptr;
  {
    std::unique_lock lock{state->gate};
    if (state->phase != PipelinePhase::Running) {
      return Status::fail(Reason::PipelineBusy);
    }
    if (state->active_step_count == 0u) {
      node::accel::detail::PreparedPipelineEvidence empty{};
      empty.check = {true, "ok"};
      empty.shared.backend = state->device->backend == Backend::Metal
                                 ? rund::AccelApi::Metal
                                 : rund::AccelApi::Vulkan;
      empty.shared.ok = true;
      empty.shared.reason = "ok";
      lock.unlock();
      completion(user, std::move(empty));
      return Status::success();
    }
    prepared = state->transactional && state->attempt_parity != 0u
                   ? &state->alternate_prepared
                   : &state->prepared;
    if (!prepared->ok) {
      return Status::fail(Reason::PipelineInvalid);
    }
  }
  const rund::AccelCheck submitted = ops->submit_pipeline(
      *state->device, *prepared, std::move(lifetime), completion, user);
  std::lock_guard lock{state->gate};
  if (!submitted.ok) {
    return Status::fail(
        project_reason(submitted.reason, Reason::BackendFailed));
  }
  state->backend_submitted = true;
  return Status::success();
}

Status finish_pipeline_on(
    const std::shared_ptr<PipelineState> &state,
    node::accel::detail::PreparedPipelineEvidence &&evidence) noexcept {
  if (!valid_pipeline(state)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::lock_guard lock{state->gate};
  if (state->phase != PipelinePhase::Running) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const DeviceOps *const ops = state->device->ops;
  if (ops == nullptr) {
    publish_pipeline_terminal(
        *state,
        PipelineTerminal{.reason = Reason::AccelProgramInvalid,
                         .verified = state->verified,
                         .failed_step = state->verified,
                         .failure_step_known = state->failure_step_known,
                         .writes_possible =
                             state->writes_possible || state->backend_submitted,
                         .publication_suppressed = !state->backend_submitted});
    return Status::fail(Reason::AccelProgramInvalid);
  }
  const PipelineOutcome outcome = finish_accel_pipeline(*state, evidence);
  if (!outcome.status) {
    publish_pipeline_terminal(
        *state,
        PipelineTerminal{
            .reason = outcome.status.reason(),
            .verified = outcome.verified,
            .failed_step = outcome.failed_step,
            .failure_step_known = outcome.failure_step_known,
            .writes_possible = outcome.writes_possible || outcome.submitted,
            .publication_suppressed = outcome.publication_suppressed});
    return outcome.status;
  }
  publish_pipeline_terminal(*state, PipelineTerminal{});
  return Status::success();
}

void record_pipeline_frame(const std::shared_ptr<PipelineState> &state,
                           const std::uint64_t bytes, const bool reused,
                           const std::uint64_t budget) noexcept {
  if (state == nullptr) {
    return;
  }
  std::lock_guard lock{state->gate};
  ::rund::detail::counter::Accumulate(state->frame_current, bytes);
  state->frame_peak = std::max(state->frame_peak, state->frame_current);
  ::rund::detail::counter::Accumulate(state->frame_bytes, bytes);
  ::rund::detail::counter::Accumulate(state->frame_reused, reused ? bytes : 0u);
  state->frame_budget = std::max(state->frame_budget, budget);
}

void release_pipeline_frame(const std::shared_ptr<PipelineState> &state,
                            const std::uint64_t bytes) noexcept {
  if (state == nullptr) {
    return;
  }
  std::lock_guard lock{state->gate};
  ::rund::detail::counter::Release(state->frame_current, bytes);
}

} // namespace rund::compute::detail
