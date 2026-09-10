#include "internal.hpp"

#include "../../../compute/job/cpu/model.hpp"
#include "../../runtime/local.hpp"

#include <utility>

namespace rund::node::compute_operation_detail {
namespace {

[[nodiscard]] std::shared_ptr<compute::detail::JobState>
JobOwner(const std::shared_ptr<void> &owner) noexcept {
  return std::static_pointer_cast<compute::detail::JobState>(owner);
}

} // namespace

[[nodiscard]] compute::Result<compute::Backend>
JobBackend(const std::shared_ptr<void> &owner) noexcept {
  return compute::detail::job_backend(JobOwner(owner));
}

[[nodiscard]] kernel::u32
JobWorkers(const std::shared_ptr<void> &owner) noexcept {
  return compute::detail::job_workers(JobOwner(owner));
}

[[nodiscard]] compute::Status
ReserveJob(const std::shared_ptr<void> &owner) noexcept {
  return compute::detail::queue_job(JobOwner(owner));
}

[[nodiscard]] compute_detail::Dispatch
SubmitJobCpu(const compute_detail::Operation &operation,
             compute_detail::TaskState &task) noexcept {
  if (task.host == nullptr) {
    return compute_detail::Dispatch::failed(
        compute::Status::fail(compute::Reason::RuntimeMissing));
  }
  const std::shared_ptr<compute::detail::JobState> job =
      JobOwner(operation.owner);
  const compute::Status submitted = compute::detail::submit_cpu_job_on(
      job, task.host->async_worker_backend, task.host->workers,
      &task.cancel_requested, &task, CpuReady,
      task.host->telemetry_level == ::rund::telemetry::Level::Trace);
  if (!submitted) {
    return compute_detail::Dispatch::failed(submitted);
  }
  return job->program->empty()
             ? compute_detail::Dispatch::accepted_without_backend()
             : compute_detail::Dispatch::backend_submitted();
}

[[nodiscard]] compute_detail::Advance
AdvanceJobCpu(const compute_detail::Operation &operation,
              compute_detail::TaskState &task) noexcept {
  if (task.host == nullptr) {
    return compute_detail::Advance::failed(
        compute::Status::fail(compute::Reason::RuntimeMissing));
  }
  compute::detail::CpuJobProgress progress =
      compute::detail::advance_cpu_job_on(
          JobOwner(operation.owner), task.host->async_worker_backend,
          &task.cancel_requested, &task, CpuReady);
  switch (progress.disposition()) {
  case compute::detail::CpuJobProgressDisposition::Failed:
    return compute_detail::Advance::failed(progress.status());
  case compute::detail::CpuJobProgressDisposition::Pending:
    return compute_detail::Advance::pending();
  case compute::detail::CpuJobProgressDisposition::Complete:
    break;
  }
  task.job_result.emplace(compute::Result<compute::detail::RunState>::success(
      std::move(progress).take_run()));
  return compute_detail::Advance::complete();
}

[[nodiscard]] compute::detail::TerminalObservation
ResultJobCpu(const compute_detail::Operation &operation,
             compute_detail::TaskState &task) noexcept {
  if (!task.job_result.has_value()) {
    return compute::detail::fail_job_terminal(
        JobOwner(operation.owner),
        compute::Status::fail(compute::Reason::CompletionInvalid),
        task.frame_bytes, CaptureTerminalProfile(task));
  }
  compute::Result<compute::detail::RunState> result =
      std::move(*task.job_result);
  task.job_result.reset();
  return compute::detail::finish_job_terminal(
      JobOwner(operation.owner), std::move(result), task.frame_bytes,
      CaptureTerminalProfile(task));
}

[[nodiscard]] compute_detail::Dispatch
SubmitJobAccel(const compute_detail::Operation &operation,
               compute_detail::TaskState &task) noexcept {
  const std::shared_ptr<compute::detail::JobState> job =
      JobOwner(operation.owner);
  const compute::Status submitted = compute::detail::submit_job_on(
      job, {}, CompleteAsync, &task, KernelTimingFor(task));
  if (!submitted) {
    return compute_detail::Dispatch::failed(submitted);
  }
  return job->program->empty()
             ? compute_detail::Dispatch::accepted_without_backend()
             : compute_detail::Dispatch::backend_submitted();
}

[[nodiscard]] compute::detail::TerminalObservation
ResultJobAccel(const compute_detail::Operation &operation,
               compute_detail::TaskState &task) noexcept {
  return ResultJobCpu(operation, task);
}

[[nodiscard]] compute::detail::TerminalObservation
FailJob(const compute_detail::Operation &operation,
        compute_detail::TaskState &task,
        const compute::Status failure) noexcept {
  return compute::detail::fail_job_terminal(JobOwner(operation.owner), failure,
                                            task.frame_bytes,
                                            CaptureTerminalProfile(task));
}

[[nodiscard]] compute::detail::TerminalObservation
CancelJob(const compute_detail::Operation &operation,
          compute_detail::TaskState &task) noexcept {
  return compute::detail::cancel_job_terminal(JobOwner(operation.owner),
                                              task.frame_bytes,
                                              CaptureTerminalProfile(task));
}

void RecordJobFrame(const std::shared_ptr<void> &owner,
                    const std::uint64_t bytes, const bool reused,
                    const std::uint64_t budget) noexcept {
  compute::detail::record_job_frame(JobOwner(owner), bytes, reused, budget);
}

} // namespace rund::node::compute_operation_detail
