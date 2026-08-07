#include <rund/counter.hpp>
#include <rund/task/stats/slots.hpp>

#include "state/model/task.hpp"
#include "state/storage.hpp"

#include "reactor/timeout.hpp"
#include "timer/store.hpp"

#include <limits>

namespace rund::node {

TimerDeadline Scheduler::MakeTimerDeadline(
    const std::chrono::nanoseconds duration) const noexcept {
  const auto now = Clock::now();
  const std::int64_t now_ns =
      state_->identity.logical_time_ns.load(std::memory_order_acquire);
  const std::int64_t duration_ns = duration.count();
  if (duration_ns > 0 &&
      now_ns > std::numeric_limits<std::int64_t>::max() - duration_ns) {
    return TimerDeadline{
        .deadline = Clock::time_point::max(),
        .deadline_ns = std::numeric_limits<std::int64_t>::max(),
    };
  }
  return TimerDeadline{
      .deadline = now + std::chrono::duration_cast<Clock::duration>(duration),
      .deadline_ns = now_ns + duration_ns,
  };
}

void Scheduler::WakeDueTimers() noexcept {
  if (state_->ready.timers.empty()) {
    return;
  }
  const auto now = Clock::now();
  TimerWait wait{};
  while (TimerStorePopDue(state_->ready.timers,
                          state_->ready.timer_wait_id_index, now, &wait)) {
    const std::int64_t previous_logical_time =
        state_->identity.logical_time_ns.load(std::memory_order_acquire);
    state_->identity.logical_time_ns.store(
        std::max(previous_logical_time, wait.deadline_ns),
        std::memory_order_release);
    RecordObservation(task::ObservationKind::TimerDue, ReasonCode::Ok,
                      wait.task_id, wait.wait_id, -1, 0, 0, wait.deadline_ns);
    (void)RecordHostEvent(::rund::host::Event{
        .kind = ::rund::host::EventKind::TimerSleep,
        .status = ::rund::host::Status::Ok,
        .task_id = wait.task_id,
        .logical_time_ns = static_cast<std::uint64_t>(wait.deadline_ns)});
    if (wait.kind == TimerWaitKind::ReactorTimeout) {
      static_cast<void>(WakeReactorTimeout(wait));
      continue;
    }
    if (wait.kind == TimerWaitKind::ReactorManyTimeout) {
      static_cast<void>(WakeReactorManyTimeout(wait));
      continue;
    }
    TaskRecord *const record = state_->Find(wait.task_id);
    if (record != nullptr && record->state == TaskState::Sleeping) {
      record->state = TaskState::Ready;
      record->lane_segment_side_exit = true;
      state_->EnqueueProgress(*record);
      Record(::rund::detail::task::OperationKind::TimerWake, ReasonCode::Ok,
             record->id, 0u, wait.wait_id, 0u, -1, 0, 0, wait.deadline_ns);
    }
  }
}

int Scheduler::TimerBoundIoPollTimeoutMs() const noexcept {
  if (state_ == nullptr || state_->ready.timers.empty()) {
    return -1;
  }
  return TimerStorePollTimeoutMs(state_->ready.timers, Clock::now());
}

::rund::detail::task::AwaitDecision
Scheduler::Sleep(const std::chrono::nanoseconds duration) noexcept {
  (void)TrapLaneOwnedSegmentPrimitive(
      duration.count() == 0 ? ::rund::detail::task::OperationKind::SleepZero
                            : ::rund::detail::task::OperationKind::TimerPark);
  EnsureCurrentCommit();
  TaskRecord *record = state_->Find(CurrentTaskId());
  if (record == nullptr || record->state != TaskState::Running) {
    ::rund::detail::task::AwaitDecision result =
        FailSleep(ReasonCode::TaskContextMissing);
    CompletePrimitiveCommit();
    return result;
  }
  if (duration.count() < 0) {
    ::rund::detail::task::AwaitDecision result =
        FailSleep(ReasonCode::TimerDurationInvalid);
    CompletePrimitiveCommit();
    return result;
  }
  if (duration.count() == 0) {
    Record(::rund::detail::task::OperationKind::SleepZero, ReasonCode::Ok,
           record->id);
    CompletePrimitiveCommit();
    return ::rund::detail::task::AwaitDecision{.status =
                                                   task::Status::success()};
  }
  if (!record->coroutine_task) {
    SetLeafFailure(*record, ReasonCode::TaskLeafPrimitiveForbidden);
    ::rund::detail::task::AwaitDecision result =
        FailSleep(ReasonCode::TaskLeafPrimitiveForbidden);
    CompletePrimitiveCommit();
    return result;
  }
  if (state_->ready.timers.size() >= state_->resources.limits.timer_capacity) {
    ::rund::detail::task::AwaitDecision result =
        FailSleep(ReasonCode::TimerCapacityExceeded);
    CompletePrimitiveCommit();
    return result;
  }
  const TimerDeadline timer_deadline = MakeTimerDeadline(duration);
  const std::uint64_t wait_id = state_->identity.next_wait_id++;
  if (!TimerStorePush(state_->ready.timers, state_->ready.timer_wait_id_index,
                      TimerWait{
                          .task_id = record->id,
                          .wait_id = wait_id,
                          .deadline = timer_deadline.deadline,
                          .deadline_ns = timer_deadline.deadline_ns,
                          .sequence = state_->identity.next_timer_sequence++,
                      })) {
    ::rund::detail::task::AwaitDecision result =
        FailSleep(ReasonCode::TimerCapacityExceeded);
    CompletePrimitiveCommit();
    return result;
  }
  ::rund::detail::counter::Accumulate(
      ::rund::detail::task::Stat(state_->evidence.metrics,
                                 ::rund::detail::task::StatSlot::Timers),
      1u);
  ++::rund::detail::task::Stat(state_->evidence.metrics,
                               ::rund::detail::task::StatSlot::Parked);
  record->state = TaskState::Sleeping;
  record->wait_id = wait_id;
  Record(::rund::detail::task::OperationKind::TimerPark, ReasonCode::Ok,
         record->id, 0u, wait_id, 0u, -1, 0, 0, timer_deadline.deadline_ns);
  record->dynamic_scope_id = CurrentScopeId();
  record->lane_segment_side_exit = true;
  ++::rund::detail::task::Stat(state_->evidence.metrics,
                               ::rund::detail::task::StatSlot::CoroutineParks);
  record->coroutine_parked = true;
  return ::rund::detail::task::AwaitDecision{.status = task::Status::success(),
                                             .suspend = true};
}

} // namespace rund::node
