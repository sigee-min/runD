#include "../runtime/local.hpp"
#include "local.hpp"

#include <rund/task/coroutine.hpp>

namespace rund::node {

struct BackendAwaiter final {
  compute_detail::TaskState *task = nullptr;

  [[nodiscard]] bool await_ready() const noexcept {
    return task == nullptr ||
           task->completion_phase.load(std::memory_order_acquire) == 3u;
  }

  [[nodiscard]] bool await_suspend(std::coroutine_handle<>) const noexcept {
    return task != nullptr && task->host != nullptr &&
           task->host->scheduler.ParkExternal(task->completion_phase,
                                              task->wake);
  }

  void await_resume() const noexcept {}
};

[[nodiscard]] runtime_detail::ComputeHostState *
BeginCoordinator(compute_detail::TaskState *const task) noexcept {
  if (task->cancel_requested.load(std::memory_order_acquire)) {
    const compute::Status status = FinishOperation(
        *task, compute::Status::fail(compute::Reason::Cancelled));
    Complete(task, status, OperationEvidence(task->operation));
    return nullptr;
  }
  runtime_detail::ComputeHostState *const host = task->host;
  if (host == nullptr) {
    const compute::Status status = FinishOperation(
        *task, compute::Status::fail(compute::Reason::RuntimeMissing));
    Complete(task, status, OperationEvidence(task->operation));
    return nullptr;
  }
  Signal(host, ::rund::TraceEvent::ComputeDispatchStarted);
  if (!host->scheduler.CurrentHandle()) {
    const compute::Status failure = FinishOperation(
        *task, compute::Status::fail(compute::Reason::CompletionInvalid));
    Complete(task, failure, OperationEvidence(task->operation));
    return nullptr;
  }
  return host;
}

task::Task<void> RunCpuCoordinator(compute_detail::TaskState *const task) {
  runtime_detail::ComputeHostState *const host = BeginCoordinator(task);
  if (host == nullptr) {
    co_return;
  }
  const compute_detail::Dispatch submitted =
      task->operation.table->submit_cpu(task->operation, *task);
  if (!submitted) {
    const compute::Status status =
        compute_detail::FinishFailure(*task, submitted.status);
    Complete(task, status, OperationEvidence(task->operation));
    co_return;
  }
  task->backend_submitted.store(submitted.backend_submitted,
                                std::memory_order_release);
  if (submitted.backend_submitted) {
    Signal(host, ::rund::TraceEvent::ComputeBackendSubmitted);
  }
  for (;;) {
    co_await BackendAwaiter{task};
    task->completion_phase.store(0u, std::memory_order_release);
    const compute_detail::Advance progress =
        task->operation.table->advance_cpu(task->operation, *task);
    if (!progress) {
      const compute::Status status =
          compute_detail::FinishFailure(*task, progress.status);
      Complete(task, status, OperationEvidence(task->operation));
      co_return;
    }
    if (progress.backend_submitted &&
        !task->backend_submitted.exchange(true, std::memory_order_acq_rel)) {
      Signal(host, ::rund::TraceEvent::ComputeBackendSubmitted);
    }
    if (!progress.complete) {
      continue;
    }
    const compute::Status status = compute_detail::FinishCpu(*task);
    Complete(task, status, OperationEvidence(task->operation));
    co_return;
  }
}

task::Task<void> RunAccelCoordinator(compute_detail::TaskState *const task) {
  runtime_detail::ComputeHostState *const host = BeginCoordinator(task);
  if (host == nullptr) {
    co_return;
  }
  const compute_detail::Dispatch submitted =
      task->operation.table->submit_accel(task->operation, *task);
  if (!submitted) {
    const compute::Status status =
        compute_detail::FinishFailure(*task, submitted.status);
    Complete(task, status, OperationEvidence(task->operation));
    co_return;
  }
  task->backend_submitted.store(submitted.backend_submitted,
                                std::memory_order_release);
  if (submitted.backend_submitted) {
    Signal(host, ::rund::TraceEvent::ComputeBackendSubmitted);
  }
  co_await BackendAwaiter{task};
  const compute::Status status = compute_detail::FinishAccel(*task);
  Complete(task, status, OperationEvidence(task->operation));
  co_return;
}

} // namespace rund::node

namespace rund::node {

void PrepareTask(compute_detail::TaskState &slot,
                 runtime_detail::ComputeHostState &host,
                 const compute_detail::Operation &operation) {
  slot.host = &host;
  slot.operation = operation;
  slot.handle = {};
  slot.completion = {};
  slot.wake = {};
  slot.status = compute::Status::fail(compute::Reason::TaskInvalid);
  slot.stats = {};
  slot.job_result.reset();
  slot.pipeline_evidence.reset();
  slot.pipeline_schedule = {};
  slot.cancel_requested.store(false, std::memory_order_relaxed);
  slot.backend_submitted.store(false, std::memory_order_relaxed);
  slot.completion_phase.store(0u, std::memory_order_relaxed);
  slot.terminal_phase.store(compute_detail::TerminalPhase::Open,
                            std::memory_order_relaxed);
  slot.submitted = false;
  slot.external_started.store(false, std::memory_order_relaxed);
  slot.retiring = false;
  slot.retired = false;
  slot.frame_bytes = 0u;
}

compute_detail::TaskState *
ClaimTask(runtime_detail::ComputeHostState &host,
          const compute_detail::Operation &operation) {
  const std::optional<std::size_t> claimed =
      host.task_slots.claim(host.tasks.size());
  if (!claimed.has_value()) {
    return nullptr;
  }
  if (*claimed == host.tasks.size()) {
    auto task = std::make_unique<compute_detail::TaskState>();
    task->slot = *claimed;
    host.tasks.push_back(std::move(task));
  }
  compute_detail::TaskState *const slot = host.tasks[*claimed].get();
  PrepareTask(*slot, host, operation);
  return slot;
}

void Retire(const std::shared_ptr<runtime_detail::ComputeHostState> &host,
            compute_detail::TaskState *const task) noexcept {
  if (task == nullptr) {
    return;
  }
  {
    std::unique_lock lock{task->mutex};
    if (task->retired) {
      return;
    }
    if (task->retiring) {
      task->retired_cv.wait(lock, [&] { return task->retired; });
      return;
    }
    if (!task->handle) {
      task->retired = true;
      task->retired_cv.notify_all();
      return;
    }
    task->retiring = true;
  }
  task::Status joined{};
  if (host != nullptr) {
    if (InSchedulerTask(*host)) {
      joined = host->scheduler.Join(&task->handle, 1u);
    } else if (task->external_started.load(std::memory_order_acquire)) {
      const task::Status completed =
          task->completion.wait == nullptr
              ? task::Status::fail(::rund::ReasonCode::TaskHandleStale)
              : task->completion.wait(task->completion.authority,
                                      task->completion.slot,
                                      task->completion.generation);
      if (!completed) {
        joined = task::Status::fail(completed.code());
      } else if (OwnsSchedulerControl(*host)) {
        joined = host->scheduler.Join(&task->handle, 1u);
      } else {
        std::lock_guard control{host->control};
        joined = host->scheduler.Join(&task->handle, 1u);
      }
    } else if (OwnsSchedulerControl(*host)) {
      joined = host->scheduler.Join(&task->handle, 1u);
    } else {
      std::lock_guard control{host->control};
      joined = host->scheduler.Join(&task->handle, 1u);
    }
  }
  std::unique_lock lock{task->mutex};
  if (!joined && task->status.error() == "compute_task_invalid") {
    task->status = compute::Status::fail(compute::Reason::CompletionInvalid);
  }
  task->retired = true;
  task->retiring = false;
  lock.unlock();
  task->retired_cv.notify_all();
}

void Release(compute_detail::TaskState *&task) noexcept {
  if (task == nullptr) {
    return;
  }
  runtime_detail::ComputeHostState *host = nullptr;
  std::size_t slot = 0u;
  {
    std::lock_guard lock{task->mutex};
    host = task->host;
    slot = task->slot;
    if (task->completion.release != nullptr) {
      task->completion.release(task->completion.authority,
                               task->completion.slot,
                               task->completion.generation);
      task->completion = {};
    }
    if (task->operation && task->operation.table->release != nullptr) {
      task->operation.table->release(task->operation.owner);
    } else {
      task->operation.owner.reset();
    }
    task->host = nullptr;
  }
  if (host != nullptr) {
    (void)host->task_slots.release(slot);
  }
  task = nullptr;
}

} // namespace rund::node

namespace rund::node::runtime_detail {

void RetireSubmitted(const std::shared_ptr<ComputeHostState> &host) noexcept {
  if (host == nullptr) {
    return;
  }
  for (std::size_t index = 0u; index < host->tasks.size(); ++index) {
    compute_detail::TaskState *task = nullptr;
    bool submitted = false;
    {
      std::lock_guard lock{host->mutex};
      if (host->task_slots.claimed(index)) {
        task = host->tasks[index].get();
        submitted = task != nullptr && task->submitted;
      }
    }
    if (submitted) {
      Retire(host, task);
    }
  }
}

void CancelSubmitted(const std::shared_ptr<ComputeHostState> &host) noexcept {
  if (host == nullptr) {
    return;
  }
  {
    std::lock_guard lock{host->mutex};
    for (std::size_t index = 0u; index < host->tasks.size(); ++index) {
      if (compute_detail::TaskState *const task = host->tasks[index].get();
          task != nullptr && host->task_slots.claimed(index)) {
        const compute_detail::CancelClaim claim =
            compute_detail::RequestCancel(task->terminal_phase);
        if (claim == compute_detail::CancelClaim::Accept ||
            claim == compute_detail::CancelClaim::Cancelled) {
          task->cancel_requested.store(true, std::memory_order_release);
        }
      }
    }
  }
}

} // namespace rund::node::runtime_detail
