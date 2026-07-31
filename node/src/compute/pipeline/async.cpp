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
         : state->transactional && state->parity != 0u
             ? state->steps[index].alternate_job
             : state->steps[index].job;
}

Status begin_pipeline_step(const std::shared_ptr<PipelineState> &state,
                           const std::size_t index) noexcept {
  if (!valid_pipeline(state) || index >= state->steps.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::lock_guard lock{state->gate};
  if (state->phase != PipelinePhase::Running || index != state->verified) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const PipelineStep &step = state->steps[index];
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
  if (state->phase != PipelinePhase::Running || index != state->verified) {
    return Status::fail(Reason::PipelineInvalid);
  }
  PipelineStep &step = state->steps[index];
  state->writes_possible = state->writes_possible || step.writes;
  Reason semantic = result.reason();
  if (result) {
    const std::shared_ptr<JobState> &job =
        state->transactional && state->parity != 0u ? step.alternate_job
                                                    : step.job;
    const Status published = publish_cpu_pipeline_views(job);
    if (published) {
      const Status consumed = consume_cpu_pipeline_step(*state, index);
      semantic = consumed.reason();
    } else {
      semantic = published.reason();
    }
  }
  const Status outcome =
      semantic == Reason::Ok ? Status::success() : Status::fail(semantic);
  capture_cpu_pipeline_step(*state, index, true);
  finish_pipeline_profile_step(*state, index);
  if (!outcome) {
    state->failure_step_known = true;
    state->stats.pipeline.verified_step_count =
        logical_verified_steps(*state, state->verified);
    state->stats.pipeline.failed_step_index = logical_step_index(*state, index);
    return outcome;
  }
  ++state->verified;
  state->stats.pipeline.verified_step_count =
      logical_verified_steps(*state, state->verified);
  return Status::success();
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
    prepared = state->transactional && state->parity != 0u
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
