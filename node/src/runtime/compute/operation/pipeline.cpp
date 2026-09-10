#include "internal.hpp"

#include "../../../compute/job/cpu/model.hpp"
#include "../../../compute/pipeline/local.hpp"
#include "../../runtime/local.hpp"

#include <utility>

namespace rund::node::compute_operation_detail {
namespace {

[[nodiscard]] std::shared_ptr<compute::detail::PipelineState>
PipelineOwner(const std::shared_ptr<void> &owner) noexcept {
  return std::static_pointer_cast<compute::detail::PipelineState>(owner);
}

[[nodiscard]] compute_detail::Advance
SubmitPipelineStep(const std::shared_ptr<compute::detail::PipelineState> &state,
                   compute_detail::TaskState &task) noexcept {
  if (task.host == nullptr) {
    return compute_detail::Advance::failed(
        compute::Status::fail(compute::Reason::CompletionInvalid));
  }
  const compute::detail::CpuPipelineSelection selected =
      compute::detail::select_cpu_pipeline_step(state, task.pipeline_schedule);
  switch (selected.disposition()) {
  case compute::detail::CpuPipelineSelectionDisposition::Failed:
    return compute_detail::Advance::failed(selected.status());
  case compute::detail::CpuPipelineSelectionDisposition::Complete:
    return compute_detail::Advance::complete();
  case compute::detail::CpuPipelineSelectionDisposition::Selected:
    break;
  }
  const std::shared_ptr<compute::detail::JobState> job = selected.job();
  const compute::Status submitted = compute::detail::submit_cpu_pipeline_job_on(
      job, task.host->async_worker_backend, task.host->workers,
      &task.cancel_requested, &task, CpuReady,
      task.host->telemetry_level == ::rund::telemetry::Level::Trace);
  if (!submitted) {
    const compute::Status completed =
        compute::detail::complete_cpu_pipeline_schedule_step(
            state, task.pipeline_schedule, submitted);
    return compute_detail::Advance::failed(completed ? submitted : completed);
  }
  const bool backend_submitted =
      job->program != nullptr && !job->program->empty();
  if (backend_submitted) {
    const compute::Status started =
        compute::detail::begin_pipeline_step(state, selected.step());
    if (!started) {
      (void)compute::detail::cancel_job(job);
      return compute_detail::Advance::failed(started);
    }
  }
  return backend_submitted ? compute_detail::Advance::backend_submitted()
                           : compute_detail::Advance::pending();
}

} // namespace

[[nodiscard]] compute::Result<compute::Backend>
PipelineBackend(const std::shared_ptr<void> &owner) noexcept {
  return compute::detail::pipeline_backend(PipelineOwner(owner));
}

[[nodiscard]] kernel::u32
PipelineWorkers(const std::shared_ptr<void> &owner) noexcept {
  return compute::detail::pipeline_workers(PipelineOwner(owner));
}

[[nodiscard]] compute::Status
ReservePipeline(const std::shared_ptr<void> &owner) noexcept {
  return compute::detail::queue_pipeline(PipelineOwner(owner));
}

[[nodiscard]] compute_detail::Dispatch
SubmitPipelineCpu(const compute_detail::Operation &operation,
                  compute_detail::TaskState &task) noexcept {
  const std::shared_ptr<compute::detail::PipelineState> state =
      PipelineOwner(operation.owner);
  const compute::Status initialized =
      compute::detail::initialize_cpu_pipeline_schedule(state,
                                                        task.pipeline_schedule);
  if (!initialized) {
    return compute_detail::Dispatch::failed(initialized);
  }
  const compute_detail::Advance submitted = SubmitPipelineStep(state, task);
  switch (submitted.disposition()) {
  case compute_detail::AdvanceDisposition::Failed:
    return compute_detail::Dispatch::failed(submitted.status());
  case compute_detail::AdvanceDisposition::Complete:
    CpuReady(&task);
    return compute_detail::Dispatch::accepted_without_backend();
  case compute_detail::AdvanceDisposition::Pending:
    return compute_detail::Dispatch::accepted_without_backend();
  case compute_detail::AdvanceDisposition::BackendSubmitted:
    return compute_detail::Dispatch::backend_submitted();
  }
  return compute_detail::Dispatch::failed(
      compute::Status::fail(compute::Reason::CompletionInvalid));
}

[[nodiscard]] compute_detail::Advance
AdvancePipelineCpu(const compute_detail::Operation &operation,
                   compute_detail::TaskState &task) noexcept {
  const std::shared_ptr<compute::detail::PipelineState> state =
      PipelineOwner(operation.owner);
  if (task.host == nullptr) {
    return compute_detail::Advance::failed(
        compute::Status::fail(compute::Reason::CompletionInvalid));
  }
  if (task.pipeline_schedule.step == compute::detail::pipeline_size(state)) {
    return compute_detail::Advance::complete();
  }
  if (task.pipeline_schedule.step > compute::detail::pipeline_size(state)) {
    return compute_detail::Advance::failed(
        compute::Status::fail(compute::Reason::CompletionInvalid));
  }
  const std::shared_ptr<compute::detail::JobState> job =
      compute::detail::pipeline_job(state, task.pipeline_schedule.step);
  const compute::detail::CpuStepProgress progress =
      compute::detail::advance_cpu_pipeline_job_on(
          job, task.host->async_worker_backend, &task.cancel_requested, &task,
          CpuReady);
  switch (progress.disposition()) {
  case compute::detail::CpuStepDisposition::Failed: {
    const compute::Status completed =
        compute::detail::complete_cpu_pipeline_schedule_step(
            state, task.pipeline_schedule, progress.status());
    return compute_detail::Advance::failed(completed ? progress.status()
                                                     : completed);
  }
  case compute::detail::CpuStepDisposition::Pending:
    return compute_detail::Advance::pending();
  case compute::detail::CpuStepDisposition::Complete:
    break;
  }
  const compute::Status advanced =
      compute::detail::complete_cpu_pipeline_schedule_step(
          state, task.pipeline_schedule, compute::Status::success());
  if (!advanced) {
    return compute_detail::Advance::failed(advanced);
  }
  return SubmitPipelineStep(state, task);
}

[[nodiscard]] compute::detail::TerminalObservation
ResultPipelineCpu(const compute_detail::Operation &operation,
                  compute_detail::TaskState &task) noexcept {
  return compute::detail::complete_cpu_pipeline_terminal(
      PipelineOwner(operation.owner), task.frame_bytes,
      CaptureTerminalProfile(task));
}

[[nodiscard]] compute_detail::Dispatch
SubmitPipelineAccel(const compute_detail::Operation &operation,
                    compute_detail::TaskState &task) noexcept {
  const std::shared_ptr<compute::detail::PipelineState> pipeline =
      PipelineOwner(operation.owner);
  const compute::Status submitted = compute::detail::submit_pipeline_on(
      pipeline, {}, CompletePipeline, &task, KernelTimingFor(task));
  if (!submitted) {
    return compute_detail::Dispatch::failed(submitted);
  }
  return pipeline->prepared.ok
             ? compute_detail::Dispatch::backend_submitted()
             : compute_detail::Dispatch::accepted_without_backend();
}

[[nodiscard]] compute::detail::TerminalObservation
ResultPipelineAccel(const compute_detail::Operation &operation,
                    compute_detail::TaskState &task) noexcept {
  if (!task.pipeline_evidence.has_value()) {
    return compute::detail::fail_pipeline_terminal(
        PipelineOwner(operation.owner),
        compute::Status::fail(compute::Reason::CompletionInvalid),
        task.frame_bytes, CaptureTerminalProfile(task));
  }
  compute::detail::TerminalObservation result =
      compute::detail::finish_pipeline_terminal_on(
          PipelineOwner(operation.owner), std::move(*task.pipeline_evidence),
          task.frame_bytes, CaptureTerminalProfile(task));
  task.pipeline_evidence.reset();
  return result;
}

[[nodiscard]] compute::detail::TerminalObservation
FailPipeline(const compute_detail::Operation &operation,
             compute_detail::TaskState &task,
             const compute::Status failure) noexcept {
  return compute::detail::fail_pipeline_terminal(PipelineOwner(operation.owner),
                                                 failure, task.frame_bytes,
                                                 CaptureTerminalProfile(task));
}

[[nodiscard]] compute::detail::TerminalObservation
CancelPipeline(const compute_detail::Operation &operation,
               compute_detail::TaskState &task) noexcept {
  return compute::detail::cancel_pipeline_terminal(
      PipelineOwner(operation.owner), task.frame_bytes,
      CaptureTerminalProfile(task));
}

void RecordPipelineFrame(const std::shared_ptr<void> &owner,
                         const std::uint64_t bytes, const bool reused,
                         const std::uint64_t budget) noexcept {
  compute::detail::record_pipeline_frame(PipelineOwner(owner), bytes, reused,
                                         budget);
}

} // namespace rund::node::compute_operation_detail
