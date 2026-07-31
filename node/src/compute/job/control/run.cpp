#include "model.hpp"

#include "../../backend.hpp"
#include "../../cpu/run/state.hpp"
#include "../cpu/model.hpp"
#include "../local.hpp"
#include <rund/counter.hpp>

#include <algorithm>
#include <memory>
#include <utility>

namespace rund::compute::detail {

Result<RunState> empty_job_run(const std::shared_ptr<JobState> &state) {
  return empty_run(state);
}

bool valid_job(const std::shared_ptr<JobState> &state) noexcept {
  if (state == nullptr || state->program == nullptr || state->inputs.empty() ||
      state->outputs.empty()) {
    return false;
  }
  return std::all_of(state->inputs.begin(), state->inputs.end(),
                     [](const auto &input) { return input != nullptr; });
}

namespace {

[[nodiscard]] Status begin_job(const std::shared_ptr<JobState> &state) {
  if (!valid_job(state)) {
    return Status::fail(Reason::RunInvalid);
  }
  std::lock_guard lock{state->gate};
  if (job_busy(state->phase)) {
    return Status::fail(Reason::JobBusy);
  }
  if (state->terminal != nullptr) {
    state->terminal->last.reset();
    state->terminal->failed_stats.reset();
  }
  state->phase = JobPhase::Running;
  return Status::success();
}

} // namespace

Status start_queued_job(const std::shared_ptr<JobState> &state) {
  if (state == nullptr) {
    return Status::fail(Reason::RunInvalid);
  }
  std::lock_guard lock{state->gate};
  if (state->phase != JobPhase::Queued) {
    return Status::fail(Reason::JobBusy);
  }
  state->phase = JobPhase::Running;
  return Status::success();
}

Status finish_job(const std::shared_ptr<JobState> &state,
                  Result<RunState> result) {
  if (state == nullptr) {
    return Status::fail(Reason::RunInvalid);
  }
  std::lock_guard lock{state->gate};
  if (!result) {
    if (state->terminal != nullptr && state->program != nullptr &&
        state->program->device != nullptr &&
        state->program->device->backend == Backend::Cpu &&
        state->cpu != nullptr && state->cpu->graph != nullptr) {
      state->terminal->failed_stats = completed_state(*state).stats;
    }
    state->failure = result.reason();
    state->phase = JobPhase::Failed;
    return Status::fail(state->failure);
  }
  if (state->terminal != nullptr) {
    state->terminal->last = std::move(result).value();
  }
  ::rund::detail::counter::Accumulate(state->run_count, 1u);
  state->phase = JobPhase::Idle;
  return Status::success();
}

Status queue_job(const std::shared_ptr<JobState> &state) {
  if (!valid_job(state)) {
    return Status::fail(Reason::RunInvalid);
  }
  std::lock_guard lock{state->gate};
  if (job_busy(state->phase)) {
    return Status::fail(Reason::JobBusy);
  }
  if (state->terminal != nullptr) {
    state->terminal->last.reset();
    state->terminal->failed_stats.reset();
  }
  state->phase = JobPhase::Queued;
  return Status::success();
}

Status cancel_job(const std::shared_ptr<JobState> &state) {
  if (state == nullptr) {
    return Status::fail(Reason::RunInvalid);
  }
  std::lock_guard lock{state->gate};
  if (state->terminal != nullptr) {
    state->terminal->last.reset();
    state->terminal->failed_stats.reset();
  }
  state->failure = Reason::Cancelled;
  state->phase = JobPhase::Failed;
  return Status::fail(Reason::Cancelled);
}

Status fail_job(const std::shared_ptr<JobState> &state, const Status failure) {
  return finish_job(state, Result<RunState>::fail(failure.reason()));
}

namespace {

[[nodiscard]] Status run_job_host(const std::shared_ptr<JobState> &state) {
  const Status started = begin_job(state);
  if (!started) {
    return started;
  }
  if (state->program->empty()) {
    return finish_job(state, empty_job_run(state));
  }
  const DeviceOps *const ops = state->program->device->ops;
  if (ops != nullptr && ops->run_job == nullptr) {
    return finish_job(state, Result<RunState>::fail(Reason::DeviceInvalid));
  }
  Result<RunState> result =
      ops == nullptr ? run_cpu_job(*state) : ops->run_job(state);
  return finish_job(state, std::move(result));
}

} // namespace

Status run_job(const std::shared_ptr<JobState> &state) {
  return run_job_host(state);
}

Status run_pipeline_job(const std::shared_ptr<JobState> &state) {
  if (!valid_job(state) || state->terminal != nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (state->program->empty()) {
    return Status::success();
  }
  return state->program->device->backend == Backend::Cpu
             ? run_cpu_pipeline_job(*state)
             : Status::fail(Reason::DeviceInvalid);
}

Status submit_job_on(const std::shared_ptr<JobState> &state,
                     std::shared_ptr<void> lifetime,
                     const JobCompletion completion,
                     void *const user) noexcept {
  const Result<Backend> backend = job_backend(state);
  if (!backend) {
    return Status::fail(backend.reason());
  }
  if (*backend == Backend::Cpu) {
    return Status::fail(Reason::NodeHostCpuSubmitInvalid);
  }
  const Status started = start_queued_job(state);
  if (!started) {
    return started;
  }
  if (state->program->empty()) {
    completion(user, empty_run(state));
    return Status::success();
  }
  const DeviceOps *const ops = state->program->device->ops;
  if (ops == nullptr || ops->submit_job == nullptr) {
    return Status::fail(Reason::DeviceInvalid);
  }
  return ops->submit_job(state, std::move(lifetime), completion, user);
}

} // namespace rund::compute::detail
