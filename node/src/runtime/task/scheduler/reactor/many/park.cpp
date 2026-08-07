#include <rund/counter.hpp>
#include <rund/task/stats/slots.hpp>

#include "park/local.hpp"

namespace rund::node {

ReactorCleanupRequest
ReadyManyParkRollbackRequest(const std::uint64_t group_id,
                             const ReasonCode reason) noexcept {
  return ReactorCleanupRequest{.wait_id = 0u,
                               .group_id = group_id,
                               .reason = reason,
                               .cancel_timeout_timer = false,
                               .remove_ready_backlog = true,
                               .cleanup_siblings = true,
                               .erase_group = true,
                               .wake_owner = false};
}

::rund::net::ready::many::Wait
ReadyManyAccess::FinishParkFailure(Scheduler &scheduler,
                                   const ReasonCode reason) noexcept {
  ::rund::net::ready::many::Wait result = FailManyCode(reason);
  scheduler.CompletePrimitiveCommit();
  return result;
}

::rund::net::ready::many::Wait ReadyManyAccess::RollbackPublishedPark(
    Scheduler &scheduler, const std::uint64_t group_id,
    const ReasonCode reason) noexcept {
  static_cast<void>(ReactorCleanupWait(
      scheduler, ReadyManyParkRollbackRequest(group_id, reason)));
  return FinishParkFailure(scheduler, reason);
}

::rund::net::ready::many::Wait
ReadyManyAccess::Park(Scheduler &scheduler, ReadyManyEntry &entry,
                      const std::optional<std::chrono::nanoseconds> timeout,
                      const ::rund::net::ready::Set ready_set) noexcept {
  SchedulerState &state = *scheduler.state_;
  const bool use_timeout = timeout.has_value();
  if (ReactorRegistrySize(state.reactor.reactor) + entry.requests.size() >
      state.resources.limits.reactor_wait_capacity) {
    return FinishParkFailure(scheduler,
                             ReasonCode::ReactorWaitCapacityExceeded);
  }
  if (use_timeout &&
      state.ready.timers.size() >= state.resources.limits.timer_capacity) {
    return FinishParkFailure(scheduler, ReasonCode::TimerCapacityExceeded);
  }
  if (use_timeout && !ReactorTimeoutReserveTimerStorage(
                         state.ready.timers, state.ready.timer_wait_id_index)) {
    return FinishParkFailure(scheduler, ReasonCode::TimerCapacityExceeded);
  }

  const std::uint64_t group_id = state.identity.next_reactor_many_group_id++;
  entry.group_id = group_id;
  const std::uint64_t timer_wait_id =
      use_timeout ? state.identity.next_wait_id++ : 0u;
  const TimerDeadline timer_deadline =
      use_timeout ? scheduler.MakeTimerDeadline(*timeout) : TimerDeadline{};
  if (!ReadyManyParkPublishGroup(state, entry, group_id, timer_wait_id,
                                 ready_set)) {
    return FinishParkFailure(scheduler,
                             ReasonCode::ReactorWaitCapacityExceeded);
  }

  if (!ReadyManyAccess::ParkRegisterWaits(scheduler, entry)) {
    return RollbackPublishedPark(
        scheduler, group_id, ReasonCode::ReactorWaitCapacityExceeded);
  }

  if (use_timeout) {
    if (!ReadyManyAccess::ParkRegisterTimeout(scheduler, entry, timer_wait_id,
                                              timer_deadline)) {
      return RollbackPublishedPark(
          scheduler, group_id, ReasonCode::TimerCapacityExceeded);
    }
  }
  ::rund::detail::counter::Accumulate(
      ::rund::detail::task::Stat(state.evidence.metrics,
                                 ::rund::detail::task::StatSlot::Parked),
      1u);
  entry.record->state = TaskState::IoBlocked;
  entry.record->wait_id = group_id;
  entry.record->wait_source_id = group_id;
  entry.record->io_result = ReasonCode::Ok;
  entry.record->io_revents = 0;
  entry.record->dynamic_scope_id = scheduler.CurrentScopeId();
  entry.record->lane_segment_side_exit = true;
  entry.record->coroutine_parked = true;
  ::rund::detail::counter::Accumulate(
      ::rund::detail::task::Stat(
          state.evidence.metrics,
          ::rund::detail::task::StatSlot::CoroutineParks),
      1u);
  if (use_timeout) {
    scheduler.Record(::rund::detail::task::OperationKind::TimerPark,
                     ReasonCode::Ok, entry.record->id, 0u, timer_wait_id, 0u,
                     -1, 0, 0, timer_deadline.deadline_ns);
  }

  return ::rund::net::ready::many::detail::Access::Suspend(group_id);
}

} // namespace rund::node
