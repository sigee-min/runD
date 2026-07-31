#include <rund/task/stats/slots.hpp>

#include "../lane/residual.hpp"
#include "../state/model/join.hpp"
#include "../state/model/task.hpp"
#include "../state/storage.hpp"
#include "../state/task/commit.hpp"

namespace rund::node {

::rund::detail::task::AwaitDecision
Scheduler::BeginJoinAwait(const task::Handle *const handles,
                          const std::size_t count) noexcept {
  (void)TrapLaneOwnedSegmentPrimitive(
      ::rund::detail::task::OperationKind::JoinPark);
  EnsureCurrentCommit();
  TaskRecord *const active = state_->Find(CurrentTaskId());
  if (active == nullptr || !active->coroutine_task) {
    if (active != nullptr) {
      SetLeafFailure(*active, ReasonCode::TaskLeafPrimitiveForbidden);
    }
    CompletePrimitiveCommit();
    return ::rund::detail::task::AwaitDecision{
        .status = task::Status::fail(ReasonCode::TaskLeafPrimitiveForbidden)};
  }
  if (count != 1u || handles == nullptr || !handles[0]) {
    const ReasonCode code = handles == nullptr || count != 1u
                                ? ReasonCode::TaskHandleInvalid
                                : handles[0].code();
    CompletePrimitiveCommit();
    return ::rund::detail::task::AwaitDecision{.status =
                                                   task::Status::fail(code)};
  }
  const std::size_t join_slot = state_->IndexFor(handles[0].id());
  const TaskRecord *const target = state_->FindAt(join_slot, handles[0].id());
  if (!Matches(handles[0], target)) {
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::JoinSlotRejections);
    CompletePrimitiveCommit();
    return ::rund::detail::task::AwaitDecision{
        .status = task::Status::fail(ReasonCode::TaskHandleUnknown)};
  }
  if (target->id == active->id) {
    CompletePrimitiveCommit();
    return ::rund::detail::task::AwaitDecision{
        .status = task::Status::fail(ReasonCode::TaskDeadlock)};
  }
  ++::rund::detail::task::Stat(state_->evidence.metrics,
                               ::rund::detail::task::StatSlot::Joins);
  if (target->state == TaskState::Completed ||
      target->state == TaskState::Failed) {
    const ReasonCode code = target->state == TaskState::Failed
                                ? target->failure_code
                                : ReasonCode::Ok;
    CompletePrimitiveCommit();
    return ::rund::detail::task::AwaitDecision{
        .status = code == ReasonCode::Ok ? task::Status::success()
                                         : task::Status::fail(code)};
  }
  if (state_->ready.join_waits.size() >=
      state_->resources.limits.task_capacity) {
    CompletePrimitiveCommit();
    return ::rund::detail::task::AwaitDecision{
        .status = task::Status::fail(ReasonCode::TaskCapacityExceeded)};
  }
  const std::uint64_t wait_id = state_->identity.next_wait_id++;
  state_->ready.join_waits.push_back(JoinWait{
      .waiter_task_id = active->id,
      .target_task_id = handles[0].id(),
      .wait_id = wait_id,
  });
  ++::rund::detail::task::Stat(state_->evidence.metrics,
                               ::rund::detail::task::StatSlot::Parked);
  active->state = TaskState::JoinBlocked;
  active->wait_id = wait_id;
  active->wait_source_id = handles[0].id();
  active->wait_result = ReasonCode::Ok;
  active->dynamic_scope_id = CurrentScopeId();
  active->lane_segment_side_exit = true;
  Record(::rund::detail::task::OperationKind::JoinPark, ReasonCode::Ok,
         active->id, handles[0].id(), wait_id);
  ++::rund::detail::task::Stat(state_->evidence.metrics,
                               ::rund::detail::task::StatSlot::CoroutineParks);
  active->coroutine_parked = true;
  return ::rund::detail::task::AwaitDecision{.status = task::Status::success(),
                                             .suspend = true};
}

task::Status Scheduler::Join(const task::Handle *const handles,
                             const std::size_t count) noexcept {
  (void)TrapLaneOwnedSegmentPrimitive(
      ::rund::detail::task::OperationKind::JoinPark);
  if (CurrentTaskId() != 0u) {
    return task::Status::fail(RejectPrimitive());
  }
  if (count != 1u) {
    return JoinManyWithSlots(handles, count);
  }
  if (handles == nullptr) {
    return FailJoin(ReasonCode::TaskHandleInvalid);
  }
  if (!handles[0]) {
    return FailJoin(handles[0].code());
  }
  const std::size_t join_slot = state_->IndexFor(handles[0].id());
  TaskRecord *const initial = state_->FindAt(join_slot, handles[0].id());
  if (!Matches(handles[0], initial)) {
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::JoinSlotRejections);
    return FailJoin(ReasonCode::TaskHandleUnknown);
  }
  if (state_->batches.lane_residual_join_owner_policy.active) [[unlikely]] {
    const auto &policy = state_->batches.lane_residual_join_owner_policy;
    if (handles[0].id() >= policy.first_task_id &&
        handles[0].id() <= policy.last_task_id &&
        initial->state == TaskState::Completed) {
      ControlCommitScope commit{*this};
      TaskRecord *const committed = state_->FindAt(join_slot, handles[0].id());
      if (Matches(handles[0], committed) &&
          committed->state == TaskState::Completed) {
        RecordLaneResidualJoinOwnerChecks(state_->evidence.metrics, 1u);
        ++::rund::detail::task::Stat(state_->evidence.metrics,
                                     ::rund::detail::task::StatSlot::Joins);
        ++::rund::detail::task::Stat(
            state_->evidence.metrics,
            ::rund::detail::task::StatSlot::RootJoinTerminalRecordReuses);
        QueueRootJoinRetireKnown(*committed, handles[0].id(), ReasonCode::Ok);
        return task::Status::success();
      }
    }
  }
  for (;;) {
    if (TerminalAt(handles, &join_slot, 1u)) {
      break;
    }
    if (RunRootSingleJoinReadyTarget(handles[0].id())) {
      continue;
    }
    if (!Step()) {
      if (TerminalAt(handles, &join_slot, 1u)) {
        break;
      }
      if (WakeDeadlockedTasks()) {
        continue;
      }
      ControlCommitScope commit{*this};
      ++::rund::detail::task::Stat(state_->evidence.metrics,
                                   ::rund::detail::task::StatSlot::Joins);
      Record(::rund::detail::task::OperationKind::JoinPark,
             ReasonCode::TaskDeadlock, CurrentTaskId(), handles[0].id());
      return FailJoin(ReasonCode::TaskDeadlock);
    }
    if (state_->batches.lane_residual_join_owner_policy.active) [[unlikely]] {
      const auto &policy = state_->batches.lane_residual_join_owner_policy;
      TaskRecord *const residual = state_->FindAt(join_slot, handles[0].id());
      if (handles[0].id() >= policy.first_task_id &&
          handles[0].id() <= policy.last_task_id && residual != nullptr &&
          Matches(handles[0], residual) &&
          residual->state == TaskState::Completed) {
        ControlCommitScope commit{*this};
        TaskRecord *const committed =
            state_->FindAt(join_slot, handles[0].id());
        if (Matches(handles[0], committed) &&
            committed->state == TaskState::Completed) {
          RecordLaneResidualJoinOwnerChecks(state_->evidence.metrics, 1u);
          ++::rund::detail::task::Stat(state_->evidence.metrics,
                                       ::rund::detail::task::StatSlot::Joins);
          ++::rund::detail::task::Stat(
              state_->evidence.metrics,
              ::rund::detail::task::StatSlot::RootJoinTerminalRecordReuses);
          QueueRootJoinRetireKnown(*committed, handles[0].id(), ReasonCode::Ok);
          return task::Status::success();
        }
      }
    }
  }
  ControlCommitScope commit{*this};
  ++::rund::detail::task::Stat(state_->evidence.metrics,
                               ::rund::detail::task::StatSlot::Joins);
  TaskRecord *const record = state_->FindAt(join_slot, handles[0].id());
  const bool matches = record != nullptr && Matches(handles[0], record);
  const ReasonCode code =
      matches ? (record->state == TaskState::Failed ? record->failure_code
                                                    : ReasonCode::Ok)
              : ReasonCode::TaskHandleUnknown;
  if (matches) {
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::RootJoinTerminalRecordReuses);
    QueueRootJoinRetireKnown(*record, handles[0].id(), code);
  } else {
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::JoinSlotRejections);
  }
  return code == ReasonCode::Ok ? task::Status::success()
                                : task::Status::fail(code);
}

} // namespace rund::node
