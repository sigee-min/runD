#include "internal.hpp"

#include "../../runtime/local.hpp"
#include "../../task/scheduler/state/model/context.hpp"

#include <rund/compute/abi/observe.hpp>
#include <rund/reason.hpp>

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

void RecordOperationFrame(const compute_detail::Operation &operation,
                          const std::uint64_t bytes, const bool reused,
                          const std::uint64_t budget) noexcept {
  if (!operation || operation.table->record_frame == nullptr) {
    return;
  }
  operation.table->record_frame(operation.owner, bytes, reused, budget);
}

[[nodiscard]] compute::detail::TerminalObservation
FinishOperation(compute_detail::TaskState &task,
                const compute::Status failure) noexcept {
  return compute_detail::FinishFailure(task, failure);
}

void Complete(compute_detail::TaskState *const task,
              compute::detail::TerminalObservation observation) noexcept {
  task->frame_bytes = 0u;
  const compute::Status status = observation.status();
  runtime_detail::ComputeHostState *const host = task->host;
  Signal(host, ::rund::TraceEvent::ComputeCompleted, status.reason());
  if (host != nullptr && host->emit != nullptr) {
    const compute::telemetry::Profile *const profile = observation.profile();
    if (profile != nullptr) {
      host->emit(host->emit_context, status, *profile);
    }
  }
  {
    std::lock_guard lock{task->mutex};
    task->status = status;
    task->stats = observation.stats();
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

namespace rund::node::compute_operation_detail {

void CompleteAsync(
    void *const raw,
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

[[nodiscard]] node::accel::detail::KernelTiming
KernelTimingFor(const compute_detail::TaskState &task) noexcept {
  if (task.host == nullptr) {
    return node::accel::detail::KernelTiming::None;
  }
  switch (task.host->telemetry_level) {
  case ::rund::telemetry::Level::Basic:
    return node::accel::detail::KernelTiming::None;
  case ::rund::telemetry::Level::Detail:
    return node::accel::detail::KernelTiming::Submission;
  case ::rund::telemetry::Level::Trace:
    return node::accel::detail::KernelTiming::Dispatch;
  }
  return node::accel::detail::KernelTiming::None;
}

[[nodiscard]] bool
CaptureTerminalProfile(const compute_detail::TaskState &task) noexcept {
  return task.host != nullptr && task.host->emit != nullptr;
}

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

} // namespace rund::node::compute_operation_detail
