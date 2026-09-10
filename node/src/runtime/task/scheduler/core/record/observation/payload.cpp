#include "local.hpp"

#include <limits>

namespace rund::node {
namespace {

[[nodiscard]] bool FitsHostPayloadCapacity(const SchedulerState &state,
                                           const std::size_t bytes) noexcept {
  const std::uint64_t current =
      state.evidence.host_payload_store.logical_bytes();
  const std::uint64_t reserved = state.evidence.host_payload_reserved_bytes;
  const std::uint64_t limit =
      state.resources.limits.host_payload_capacity_bytes;
  const std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
  if (current > max - reserved) {
    return false;
  }
  const std::uint64_t used = current + reserved;
  const std::uint64_t requested = static_cast<std::uint64_t>(bytes);
  if (requested > max - used) {
    return false;
  }
  return used + requested <= limit;
}

} // namespace

bool Scheduler::CapturesNetIngress() const noexcept {
  return state_->evidence.input_capture_active.load(
             std::memory_order_acquire) &&
         state_->evidence.host_payload_store.CapturesIngress();
}

bool Scheduler::ReserveHostPayloadCapacity(const std::size_t bytes) noexcept {
  std::lock_guard lock{state_->evidence.mutex};
  state_->RequireSequencer();
  if (!FitsHostPayloadCapacity(*state_, bytes)) {
    return false;
  }
  state_->evidence.host_payload_reserved_bytes +=
      static_cast<std::uint64_t>(bytes);
  return true;
}

void Scheduler::ReleaseHostPayloadCapacity(const std::size_t bytes) noexcept {
  std::lock_guard lock{state_->evidence.mutex};
  state_->RequireSequencer();
  const std::uint64_t reserved = static_cast<std::uint64_t>(bytes);
  if (reserved > state_->evidence.host_payload_reserved_bytes) {
    state_->evidence.host_payload_reserved_bytes = 0u;
    return;
  }
  state_->evidence.host_payload_reserved_bytes -= reserved;
}

void Scheduler::FailCurrentTaskOrScheduler(const ReasonCode code) noexcept {
  std::lock_guard lock{state_->evidence.mutex};
  const char *const reason = ReasonString(code);
  const std::uint64_t task_id = CurrentTaskId();
  if (task_id != 0u) {
    if (TaskRecord *const record = state_->Find(task_id); record != nullptr) {
      const TaskState previous_state = record->state;
      record->state = TaskState::Failed;
      record->failure_code = code;
      record->lane_segment_side_exit = true;
      if (previous_state != TaskState::Running &&
          previous_state != TaskState::Completed &&
          previous_state != TaskState::Failed) {
        ++::rund::detail::task::Stat(state_->evidence.metrics,
                                     ::rund::detail::task::StatSlot::Failed);
        RecordTerminalBatch(::rund::detail::task::OperationKind::Fail, code,
                            record->id);
        WakeJoinWaiters(record->id, code);
        DestroyTask(*record);
      }
    }
    return;
  }
  state_->identity.host_replay_failed = true;
  state_->identity.host_replay_reason = reason;
  state_->identity.host_replay_payload_failed = true;
  state_->identity.host_replay_payload_reason = reason;
}

ReasonCode Scheduler::RecordHostPayloadForCommittedEvent(
    const HostEventCommitResult &commit, const ::rund::host::EventKind kind,
    const replay_detail::payload::Capture &payload) noexcept {
  ReasonCode failure = ReasonCode::Ok;
  {
    std::lock_guard lock{state_->evidence.mutex};
    state_->RequireSequencer();
    if (!commit.ok()) {
      return commit.code();
    }
    if (!commit.retained()) {
      return ReasonCode::Ok;
    }
    if (!payload || !FitsHostPayloadCapacity(*state_, payload.bytes().size())) {
      failure = ReasonCode::TaskCapacityExceeded;
    } else {
      try {
        if (!state_->evidence.host_payload_store.Append(commit.sequence(), kind,
                                                        payload)) {
          failure = ReasonCode::HostReplayPayloadMismatch;
        }
      } catch (...) {
        failure = ReasonCode::HostReplayPayloadMismatch;
      }
    }
  }
  if (failure != ReasonCode::Ok) {
    FailCurrentTaskOrScheduler(failure);
  }
  return failure;
}

} // namespace rund::node
