#include "local.hpp"
#include "../runtime/local.hpp"

#include "../../compute/pipeline/local.hpp"
#include "../task/scheduler/state/model/context.hpp"

#include <rund/reason.hpp>
#include <rund/compute/abi/observe.hpp>

#include <utility>

namespace rund::node {

void Signal(runtime_detail::ComputeHostState *const host,
            const ::rund::TraceEvent event,
            const compute::Reason reason) noexcept {
  if (host != nullptr && host->signal != nullptr) {
    host->signal(host->signal_context, event,
                 ::rund::TraceCode::compute(reason));
  }
}

[[nodiscard]] bool
InSchedulerTask(const runtime_detail::ComputeHostState &host) noexcept {
  return active_scheduler_context != nullptr &&
         active_scheduler_context->scheduler == &host.scheduler &&
         active_scheduler_context->task_id != 0u;
}

[[nodiscard]] bool
OwnsSchedulerControl(const runtime_detail::ComputeHostState &host) noexcept {
  return Scheduler::Active() == &host.scheduler;
}

[[nodiscard]] compute::Reason
SpawnReason(const ::rund::ReasonCode reason) noexcept {
  switch (reason) {
  case ::rund::ReasonCode::TaskCapacityExceeded:
    return compute::Reason::TaskCapacity;
  case ::rund::ReasonCode::ReadyQueueCapacityExceeded:
    return compute::Reason::ReadyQueueCapacity;
  case ::rund::ReasonCode::TaskCompletionCapacity:
    return compute::Reason::TaskCompletionCapacity;
  case ::rund::ReasonCode::TaskFrameCapacity:
    return compute::Reason::TaskFrameCapacity;
  case ::rund::ReasonCode::TaskFrameTooLarge:
    return compute::Reason::TaskFrameTooLarge;
  case ::rund::ReasonCode::TaskStateTransitionInvalid:
    return compute::Reason::CompletionInvalid;
  case ::rund::ReasonCode::TaskInvalid:
  case ::rund::ReasonCode::TaskFrameNotConfigured:
  case ::rund::ReasonCode::TaskFrameLimitsInvalid:
  case ::rund::ReasonCode::TaskFrameAlignment:
  case ::rund::ReasonCode::TaskFrameRuntimeMissing:
  case ::rund::ReasonCode::TaskFrameRuntimeMismatch:
  case ::rund::ReasonCode::Ok:
    return compute::Reason::TaskInvalid;
  default:
    return compute::Reason::ReasonInvalid;
  }
}

[[nodiscard]] compute::Result<compute::Backend>
OperationBackend(const compute_detail::Operation &operation) noexcept {
  return !operation || operation.table->backend == nullptr
             ? compute::Result<compute::Backend>::fail(
                   compute::Reason::TaskInvalid)
             : operation.table->backend(operation.owner);
}

[[nodiscard]] kernel::u32
OperationWorkers(const compute_detail::Operation &operation) noexcept {
  return !operation || operation.table->workers == nullptr
             ? 0u
             : operation.table->workers(operation.owner);
}

[[nodiscard]] compute::Status
ReserveOperation(const compute_detail::Operation &operation) noexcept {
  return !operation || operation.table->reserve == nullptr
             ? compute::Status::fail(compute::Reason::TaskInvalid)
             : operation.table->reserve(operation.owner);
}

[[nodiscard]] compute::Stats
OperationEvidence(const compute_detail::Operation &operation) noexcept {
  return !operation || operation.table->evidence == nullptr
             ? compute::Stats{}
             : operation.table->evidence(operation.owner);
}

void RecordOperationFrame(
    const compute_detail::Operation &operation,
    const std::uint64_t bytes, const bool reused,
    const std::uint64_t budget) noexcept {
  if (!operation || operation.table->record_frame == nullptr) {
    return;
  }
  operation.table->record_frame(operation.owner, bytes, reused, budget);
}

void ReleaseOperationFrame(
    const compute_detail::Operation &operation,
    const std::uint64_t bytes) noexcept {
  if (!operation || operation.table->release_frame == nullptr) {
    return;
  }
  operation.table->release_frame(operation.owner, bytes);
}

[[nodiscard]] compute::Status
FinishOperation(compute_detail::TaskState &task,
                const compute::Status failure) noexcept {
  return compute_detail::FinishFailure(task, failure);
}

void Complete(compute_detail::TaskState *const task,
              const compute::Status &status,
              const compute::Stats &stats) noexcept {
  ReleaseOperationFrame(task->operation, task->frame_bytes);
  task->frame_bytes = 0u;
  runtime_detail::ComputeHostState *const host = task->host;
  Signal(host, ::rund::TraceEvent::ComputeCompleted, status.reason());
  if (host != nullptr && host->emit != nullptr) {
    host->emit(host->emit_context, status, stats);
  }
  {
    std::lock_guard lock{task->mutex};
    task->status = status;
    task->stats = stats;
  }
  if (host != nullptr) {
    std::lock_guard lock{host->mutex};
    if (host->outstanding != 0u) {
      --host->outstanding;
    }
  }
  compute_detail::MarkComplete(task->terminal_phase);
  if (host != nullptr) {
    host->drained.notify_all();
  }
}

} // namespace rund::node

namespace rund::node {
namespace {

void CompleteAsync(void *const raw,
                   compute::Result<compute::detail::RunState> result) noexcept {
  auto *const task = static_cast<compute_detail::TaskState *>(raw);
  task->job_result.emplace(std::move(result));
  const std::uint8_t prior =
      task->completion_phase.exchange(3u, std::memory_order_acq_rel);
  if (prior == 2u && task->host != nullptr) {
    (void)task->host->scheduler.WakeExternal(task->wake);
  }
}

void CompletePipeline(
    void *const raw,
    node::accel::detail::PreparedPipelineEvidence &&evidence) noexcept {
  auto *const task = static_cast<compute_detail::TaskState *>(raw);
  task->pipeline_evidence.emplace(std::move(evidence));
  const std::uint8_t prior =
      task->completion_phase.exchange(3u, std::memory_order_acq_rel);
  if (prior == 2u && task->host != nullptr) {
    (void)task->host->scheduler.WakeExternal(task->wake);
  }
}

void CpuReady(void *raw) noexcept;

[[nodiscard]] std::shared_ptr<compute::detail::JobState>
JobOwner(const std::shared_ptr<void> &owner) noexcept {
  return std::static_pointer_cast<compute::detail::JobState>(owner);
}

[[nodiscard]] std::shared_ptr<compute::detail::PipelineState>
PipelineOwner(const std::shared_ptr<void> &owner) noexcept {
  return std::static_pointer_cast<compute::detail::PipelineState>(owner);
}

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
    return {.status =
                compute::Status::fail(compute::Reason::RuntimeMissing)};
  }
  const std::shared_ptr<compute::detail::JobState> job =
      JobOwner(operation.owner);
  return {.status = compute::detail::submit_cpu_job_on(
              job, task.host->async_worker_backend, task.host->workers,
              &task.cancel_requested, &task, CpuReady),
          .backend_submitted = job != nullptr && job->program != nullptr &&
                               !job->program->empty()};
}

[[nodiscard]] compute_detail::Advance
AdvanceJobCpu(const compute_detail::Operation &operation,
              compute_detail::TaskState &task) noexcept {
  if (task.host == nullptr) {
    return {.status =
                compute::Status::fail(compute::Reason::RuntimeMissing)};
  }
  compute::detail::CpuJobProgress progress =
      compute::detail::advance_cpu_job_on(
          JobOwner(operation.owner), task.host->async_worker_backend,
          &task.cancel_requested, &task, CpuReady);
  if (!progress) {
    return {.status = progress.status};
  }
  if (!progress.complete()) {
    return {};
  }
  task.job_result.emplace(
      compute::Result<compute::detail::RunState>::success(
          std::move(*progress.run)));
  return {.complete = true};
}

[[nodiscard]] compute::Status
ResultJobCpu(const compute_detail::Operation &operation,
             compute_detail::TaskState &task) noexcept {
  if (!task.job_result.has_value()) {
    return compute::detail::fail_job(
        JobOwner(operation.owner),
        compute::Status::fail(compute::Reason::CompletionInvalid));
  }
  compute::Result<compute::detail::RunState> result =
      std::move(*task.job_result);
  task.job_result.reset();
  return compute::detail::finish_job(JobOwner(operation.owner),
                                     std::move(result));
}

[[nodiscard]] compute_detail::Dispatch
SubmitJobAccel(const compute_detail::Operation &operation,
               compute_detail::TaskState &task) noexcept {
  const std::shared_ptr<compute::detail::JobState> job =
      JobOwner(operation.owner);
  return {.status = compute::detail::submit_job_on(job, {}, CompleteAsync,
                                                   &task),
          .backend_submitted = job != nullptr && job->program != nullptr &&
                               !job->program->empty()};
}

[[nodiscard]] compute::Status
ResultJobAccel(const compute_detail::Operation &operation,
               compute_detail::TaskState &task) noexcept {
  return ResultJobCpu(operation, task);
}

[[nodiscard]] compute::Status FailJob(const std::shared_ptr<void> &owner,
                                      const compute::Status failure) noexcept {
  return compute::detail::fail_job(JobOwner(owner), failure);
}

[[nodiscard]] compute::Status
CancelJob(const std::shared_ptr<void> &owner) noexcept {
  return compute::detail::cancel_job(JobOwner(owner));
}

[[nodiscard]] compute::Stats
JobEvidence(const std::shared_ptr<void> &owner) noexcept {
  return compute::detail::job_stats(JobOwner(owner));
}

void RecordJobFrame(const std::shared_ptr<void> &owner,
                    const std::uint64_t bytes, const bool reused,
                    const std::uint64_t budget) noexcept {
  compute::detail::record_job_frame(JobOwner(owner), bytes, reused, budget);
}

void ReleaseJobFrame(const std::shared_ptr<void> &owner,
                     const std::uint64_t bytes) noexcept {
  compute::detail::release_job_frame(JobOwner(owner), bytes);
}

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

struct PipelineStepDispatch final {
  compute::Status status{compute::Status::success()};
  bool complete{};
  bool backend_submitted{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(status);
  }
};

[[nodiscard]] PipelineStepDispatch
SubmitPipelineStep(const std::shared_ptr<compute::detail::PipelineState> &state,
                   compute_detail::TaskState &task) noexcept {
  if (task.host == nullptr) {
    return {.status =
                compute::Status::fail(compute::Reason::CompletionInvalid)};
  }
  const compute::detail::CpuPipelineSelection selected =
      compute::detail::select_cpu_pipeline_step(state,
                                                task.pipeline_schedule);
  if (!selected) {
    return {.status = selected.status};
  }
  if (selected.complete) {
    return {.complete = true};
  }
  const std::shared_ptr<compute::detail::JobState> job =
      selected.job;
  const compute::Status submitted =
      compute::detail::submit_cpu_pipeline_job_on(
      job, task.host->async_worker_backend, task.host->workers,
      &task.cancel_requested, &task, CpuReady);
  if (!submitted) {
    const compute::Status completed =
        compute::detail::complete_cpu_pipeline_schedule_step(
            state, task.pipeline_schedule, submitted);
    return {.status = completed ? submitted : completed};
  }
  const bool backend_submitted = job != nullptr && job->program != nullptr &&
                                 !job->program->empty();
  if (backend_submitted) {
    const compute::Status started =
        compute::detail::begin_pipeline_step(state, selected.step);
    if (!started) {
      (void)compute::detail::cancel_job(job);
      return {.status = started};
    }
  }
  return {.backend_submitted = backend_submitted};
}

[[nodiscard]] compute_detail::Dispatch
SubmitPipelineCpu(const compute_detail::Operation &operation,
                  compute_detail::TaskState &task) noexcept {
  const std::shared_ptr<compute::detail::PipelineState> state =
      PipelineOwner(operation.owner);
  const compute::Status initialized =
      compute::detail::initialize_cpu_pipeline_schedule(
          state, task.pipeline_schedule);
  if (!initialized) {
    return {.status = initialized};
  }
  const PipelineStepDispatch submitted = SubmitPipelineStep(state, task);
  if (!submitted) {
    return {.status = submitted.status};
  }
  if (submitted.complete) {
    CpuReady(&task);
  }
  return {.backend_submitted = submitted.backend_submitted};
}

[[nodiscard]] compute_detail::Advance
AdvancePipelineCpu(const compute_detail::Operation &operation,
                   compute_detail::TaskState &task) noexcept {
  const std::shared_ptr<compute::detail::PipelineState> state =
      PipelineOwner(operation.owner);
  if (task.host == nullptr) {
    return {.status =
                compute::Status::fail(compute::Reason::CompletionInvalid)};
  }
  if (task.pipeline_schedule.step == compute::detail::pipeline_size(state)) {
    return {.complete = true};
  }
  if (task.pipeline_schedule.step >
      compute::detail::pipeline_size(state)) {
    return {.status =
                compute::Status::fail(compute::Reason::CompletionInvalid)};
  }
  const std::shared_ptr<compute::detail::JobState> job =
      compute::detail::pipeline_job(state, task.pipeline_schedule.step);
  compute::detail::CpuPipelineProgress progress =
      compute::detail::advance_cpu_pipeline_job_on(
          job, task.host->async_worker_backend, &task.cancel_requested, &task,
          CpuReady);
  if (!progress) {
    const compute::Status completed =
        compute::detail::complete_cpu_pipeline_schedule_step(
            state, task.pipeline_schedule, progress.status);
    return {.status = completed ? progress.status : completed};
  }
  if (!progress.complete()) {
    return {};
  }
  const compute::Status advanced =
      compute::detail::complete_cpu_pipeline_schedule_step(
          state, task.pipeline_schedule, compute::Status::success());
  if (!advanced) {
    return {.status = advanced};
  }
  const PipelineStepDispatch submitted = SubmitPipelineStep(state, task);
  if (!submitted) {
    return {.status = submitted.status};
  }
  return {.complete = submitted.complete,
          .backend_submitted = submitted.backend_submitted};
}

[[nodiscard]] compute::Status
ResultPipelineCpu(const compute_detail::Operation &operation,
                  compute_detail::TaskState &) noexcept {
  return compute::detail::complete_cpu_pipeline(
      PipelineOwner(operation.owner));
}

[[nodiscard]] compute_detail::Dispatch
SubmitPipelineAccel(const compute_detail::Operation &operation,
                    compute_detail::TaskState &task) noexcept {
  const std::shared_ptr<compute::detail::PipelineState> pipeline =
      PipelineOwner(operation.owner);
  return {.status = compute::detail::submit_pipeline_on(
              pipeline, {}, CompletePipeline, &task),
          .backend_submitted = pipeline != nullptr && pipeline->prepared.ok};
}

[[nodiscard]] compute::Status
ResultPipelineAccel(const compute_detail::Operation &operation,
                    compute_detail::TaskState &task) noexcept {
  if (!task.pipeline_evidence.has_value()) {
    return compute::detail::fail_pipeline(
        PipelineOwner(operation.owner),
        compute::Status::fail(compute::Reason::CompletionInvalid));
  }
  const compute::Status result = compute::detail::finish_pipeline_on(
      PipelineOwner(operation.owner), std::move(*task.pipeline_evidence));
  task.pipeline_evidence.reset();
  return result;
}

[[nodiscard]] compute::Status
FailPipeline(const std::shared_ptr<void> &owner,
             const compute::Status failure) noexcept {
  return compute::detail::fail_pipeline(PipelineOwner(owner), failure);
}

[[nodiscard]] compute::Status
CancelPipeline(const std::shared_ptr<void> &owner) noexcept {
  return compute::detail::cancel_pipeline(PipelineOwner(owner));
}

[[nodiscard]] compute::Stats
PipelineEvidence(const std::shared_ptr<void> &owner) noexcept {
  return compute::detail::pipeline_stats(PipelineOwner(owner));
}

void RecordPipelineFrame(const std::shared_ptr<void> &owner,
                         const std::uint64_t bytes, const bool reused,
                         const std::uint64_t budget) noexcept {
  compute::detail::record_pipeline_frame(PipelineOwner(owner), bytes, reused,
                                         budget);
}

void ReleasePipelineFrame(const std::shared_ptr<void> &owner,
                          const std::uint64_t bytes) noexcept {
  compute::detail::release_pipeline_frame(PipelineOwner(owner), bytes);
}

void ReleaseOwner(std::shared_ptr<void> &owner) noexcept { owner.reset(); }

const compute_detail::OperationTable JobOperationTable{
    .backend = JobBackend,
    .workers = JobWorkers,
    .reserve = ReserveJob,
    .submit_cpu = SubmitJobCpu,
    .advance_cpu = AdvanceJobCpu,
    .result_cpu = ResultJobCpu,
    .submit_accel = SubmitJobAccel,
    .result_accel = ResultJobAccel,
    .fail = FailJob,
    .cancel = CancelJob,
    .evidence = JobEvidence,
    .record_frame = RecordJobFrame,
    .release_frame = ReleaseJobFrame,
    .release = ReleaseOwner,
};

const compute_detail::OperationTable PipelineOperationTable{
    .backend = PipelineBackend,
    .workers = PipelineWorkers,
    .reserve = ReservePipeline,
    .submit_cpu = SubmitPipelineCpu,
    .advance_cpu = AdvancePipelineCpu,
    .result_cpu = ResultPipelineCpu,
    .submit_accel = SubmitPipelineAccel,
    .result_accel = ResultPipelineAccel,
    .fail = FailPipeline,
    .cancel = CancelPipeline,
    .evidence = PipelineEvidence,
    .record_frame = RecordPipelineFrame,
    .release_frame = ReleasePipelineFrame,
    .release = ReleaseOwner,
};

void CpuReady(void *const raw) noexcept {
  auto *const task = static_cast<compute_detail::TaskState *>(raw);
  if (task == nullptr) {
    return;
  }
  const std::uint8_t prior =
      task->completion_phase.exchange(3u, std::memory_order_acq_rel);
  if (prior == 2u && task->host != nullptr) {
    (void)task->host->scheduler.WakeExternal(task->wake);
  }
}

} // namespace

compute_detail::Operation compute_detail::make_job(
    std::shared_ptr<compute::detail::JobState> state) noexcept {
  return Operation{.table = &JobOperationTable, .owner = std::move(state)};
}

compute_detail::Operation compute_detail::make_pipeline(
    std::shared_ptr<compute::detail::PipelineState> state) noexcept {
  return Operation{.table = &PipelineOperationTable,
                   .owner = std::move(state)};
}

compute_detail::Operation
compute_detail::make_operation(std::shared_ptr<void> owner,
                               const void *const table) noexcept {
  return Operation{.table = static_cast<const OperationTable *>(table),
                   .owner = std::move(owner)};
}

} // namespace rund::node
