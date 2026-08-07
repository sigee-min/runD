#include <rund/counter.hpp>
#include <rund/task/stats/slots.hpp>

#include "../../reactor/apply/policy.hpp"
#include "../../state/model/task.hpp"
#include "../../state/storage.hpp"
#include "../../state/task/commit.hpp"

#include <deque>

namespace rund::node {

task::Status Scheduler::JoinManyWithSlots(const task::Handle *const handles,
                                          const std::size_t count) noexcept {
  std::size_t empty_join_slot = SchedulerState::kInvalidTaskIndex;
  std::size_t *join_slots = &empty_join_slot;
  if (count != 0u) {
    if (count > state_->resources.limits.task_capacity) {
      return FailJoin(ReasonCode::TaskCapacityExceeded);
    }
    if (count > state_->resources.join_slots.size()) {
      try {
        state_->resources.join_slots.resize(count);
      } catch (...) {
        return FailJoin(ReasonCode::TaskCapacityExceeded);
      }
    }
    join_slots = state_->resources.join_slots.data();
  }
  return JoinManyWithProvidedSlots(handles, count, join_slots);
}

task::Status
Scheduler::JoinManyWithProvidedSlots(const task::Handle *const handles,
                                     const std::size_t count,
                                     std::size_t *join_slots) noexcept {
  if (handles == nullptr && count != 0u) {
    return FailJoin(ReasonCode::TaskHandleInvalid);
  }
  std::size_t empty_join_slot = SchedulerState::kInvalidTaskIndex;
  if (count == 0u) {
    join_slots = &empty_join_slot;
  } else if (join_slots == nullptr) {
    return FailJoin(ReasonCode::TaskHandleInvalid);
  }
  for (std::size_t index = 0u; index < count; ++index) {
    join_slots[index] = SchedulerState::kInvalidTaskIndex;
  }
  if (CurrentTaskId() != 0u) {
    return FailJoin(RejectPrimitive());
  }
  for (std::size_t index = 0u; index < count; ++index) {
    if (!handles[index]) {
      return FailJoin(handles[index].code());
    }
    const std::size_t slot = state_->IndexFor(handles[index].id());
    const TaskRecord *const record = state_->FindAt(slot, handles[index].id());
    if (!Matches(handles[index], record)) {
      ++::rund::detail::task::Stat(
          state_->evidence.metrics,
          ::rund::detail::task::StatSlot::JoinSlotRejections);
      return FailJoin(ReasonCode::TaskHandleUnknown);
    }
    join_slots[index] = slot;
  }
  {
    ControlCommitScope commit{*this};
    ::rund::detail::counter::Accumulate(
        ::rund::detail::task::Stat(state_->evidence.metrics,
                                   ::rund::detail::task::StatSlot::Joins),
        1u);
    state_->batches.root_single_join_streak = 0u;
    if (state_->batches.root_single_join_session_lane_active) {
      ++::rund::detail::task::Stat(
          state_->evidence.metrics,
          ::rund::detail::task::StatSlot::RootSingleJoinSessionLaneResets);
    }
    state_->batches.root_single_join_session_lane_active = false;
    state_->batches.root_single_join_session_lane = 0u;
  }
  if (count > 1u) {
    ReactorApplyBatchScope reactor_apply_batch{state_->reactor.reactor};
    for (;;) {
      if (TerminalAt(handles, join_slots, count)) {
        break;
      }
      const ReadyPick ready = PopSubmittableReady(0u);
      if (ready.activity) {
        continue;
      }
      if (ready.id == 0u || !DispatchReadyTask(ready.id, 0u)) {
        break;
      }
    }
    (void)DrainReadyReactor(0, true);
  }
  for (;;) {
    if (TerminalAt(handles, join_slots, count)) {
      break;
    }
    if (!Step()) {
      if (TerminalAt(handles, join_slots, count)) {
        break;
      }
      if (WakeDeadlockedTasks()) {
        continue;
      }
      return FailJoin(ReasonCode::TaskDeadlock);
    }
  }
  ControlCommitScope commit{*this};
  const ReasonCode code = FailureCodeAt(handles, join_slots, count);
  for (std::size_t index = 0u; index < count; ++index) {
    TaskRecord *const record =
        state_->FindAt(join_slots[index], handles[index].id());
    if (!Matches(handles[index], record) || record == nullptr) {
      ++::rund::detail::task::Stat(
          state_->evidence.metrics,
          ::rund::detail::task::StatSlot::JoinSlotRejections);
      continue;
    }
    QueueRootJoinRetireKnown(*record, handles[index].id(), code);
  }
  return code == ReasonCode::Ok ? task::Status::success()
                                : task::Status::fail(code);
}

} // namespace rund::node
