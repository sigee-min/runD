#include "../state/model/task.hpp"
#include "../state/storage/check.hpp"
#include "../state/storage.hpp"

namespace rund::node {

bool Scheduler::TerminalAt(const task::Handle *handles,
                           const std::size_t *slots,
                           std::size_t count) const noexcept {
  std::lock_guard lock{state_->evidence.mutex};
  state_->RequireSequencer();
  for (std::size_t index = 0u; index < count; ++index) {
    const TaskRecord *const record =
        state_->FindAt(slots[index], handles[index].id());
    if (!Matches(handles[index], record) || record->quantum_active ||
        (record->state != TaskState::Completed &&
         record->state != TaskState::Failed)) {
      return false;
    }
  }
  return true;
}

bool SchedulerState::ScopeTerminal(std::uint64_t scope_id) const noexcept {
  std::lock_guard lock{evidence.mutex};
  RequireSequencer();
  for (const TaskRecord &record : ready.records) {
    if (record.scope_id == scope_id &&
        (record.quantum_active || record.state != TaskState::Completed) &&
        (record.quantum_active || record.state != TaskState::Failed)) {
      return false;
    }
  }
  return true;
}

ReasonCode Scheduler::FailureCodeAt(const task::Handle *handles,
                                    const std::size_t *slots,
                                    std::size_t count) const noexcept {
  std::lock_guard lock{state_->evidence.mutex};
  state_->RequireSequencer();
  for (std::size_t index = 0u; index < count; ++index) {
    const TaskRecord *const record =
        state_->FindAt(slots[index], handles[index].id());
    if (Matches(handles[index], record) && !record->quantum_active &&
        record->state == TaskState::Failed) {
      return record->failure_code;
    }
  }
  return ReasonCode::Ok;
}

ReasonCode
SchedulerState::ScopeFailureCode(std::uint64_t scope_id) const noexcept {
  std::lock_guard lock{evidence.mutex};
  RequireSequencer();
  if (plan.failure != ReasonCode::Ok) {
    return plan.failure;
  }
  if (identity.host_replay_payload_failed) {
    return ReasonCode::HostReplayPayloadMismatch;
  }
  if (identity.host_replay_failed) {
    return ReasonCode::HostReplayEventMismatch;
  }
  for (const TaskRecord &record : ready.records) {
    if (record.scope_id == scope_id && !record.quantum_active &&
        record.state == TaskState::Failed) {
      return record.failure_code;
    }
  }
  return ReasonCode::Ok;
}

ReasonCode SchedulerState::FirstFailureCode() const noexcept {
  std::lock_guard lock{evidence.mutex};
  RequireSequencer();
  if (plan.failure != ReasonCode::Ok) {
    return plan.failure;
  }
  if (identity.host_replay_payload_failed) {
    return ReasonCode::HostReplayPayloadMismatch;
  }
  if (identity.host_replay_failed) {
    return ReasonCode::HostReplayEventMismatch;
  }
  for (const TaskRecord &record : ready.records) {
    if (!record.quantum_active && record.state == TaskState::Failed) {
      return record.failure_code;
    }
  }
  return ReasonCode::Ok;
}

} // namespace rund::node
