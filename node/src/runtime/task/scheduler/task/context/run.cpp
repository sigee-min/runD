#include <rund/task/stats/slots.hpp>

#include "local.hpp"

#include <cstdint>

namespace rund::node {

thread_local SchedulerThreadContext *active_scheduler_context = nullptr;

std::uint64_t Scheduler::CurrentTaskId() const noexcept {
  if (active_scheduler_context != nullptr &&
      active_scheduler_context->scheduler == this) {
    return active_scheduler_context->task_id;
  }
  return state_->identity.active_task_id;
}

bool Scheduler::CurrentTaskIsCoroutine() const noexcept {
  const SchedulerThreadContext *const context = active_scheduler_context;
  const auto *const record =
      context != nullptr && context->scheduler == this
          ? static_cast<const TaskRecord *>(context->record)
          : nullptr;
  return record != nullptr && record->coroutine_task;
}

std::uint64_t Scheduler::CurrentScopeId() const noexcept {
  if (active_scheduler_context != nullptr &&
      active_scheduler_context->scheduler == this) {
    return active_scheduler_context->scope_id;
  }
  return state_->identity.active_scope_id;
}

void Scheduler::SetCurrentScopeId(const std::uint64_t scope_id) noexcept {
  if (active_scheduler_context != nullptr &&
      active_scheduler_context->scheduler == this) {
    active_scheduler_context->scope_id = scope_id;
    return;
  }
  state_->identity.active_scope_id = scope_id;
}

TaskRecord *Scheduler::PrepareLaneQuantum(const std::uint64_t id) noexcept {
  TaskRecord *record = state_->Find(id);
  if (record == nullptr) {
    return nullptr;
  }
  return PrepareLaneQuantum(*record);
}

TaskRecord *Scheduler::PrepareLaneQuantum(TaskRecord &record) noexcept {
  std::lock_guard evidence_lock{state_->evidence.mutex};
  if (record.state == TaskState::Completed ||
      record.state == TaskState::Failed) {
    return nullptr;
  }
  if (record.coroutine_task) {
    if (record.coroutine_frame == nullptr) {
      record.state = TaskState::Failed;
      record.failure_code = ReasonCode::TaskInvalid;
      return nullptr;
    }
    record.quantum_active = true;
    record.state = TaskState::Running;
    if (record.coroutine_parked) {
      record.coroutine_parked = false;
      ++::rund::detail::task::Stat(
          state_->evidence.metrics,
          ::rund::detail::task::StatSlot::CoroutineWakes);
    }
    return &record;
  }
  record.quantum_active = true;
  record.state = TaskState::Running;
  return &record;
}

void Scheduler::FinishQuantum(TaskRecord &record) noexcept {
  std::lock_guard evidence_lock{state_->evidence.mutex};
  record.quantum_active = false;
}

} // namespace rund::node
