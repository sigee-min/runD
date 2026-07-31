#include <rund/task/stats/slots.hpp>

#include "local.hpp"

namespace rund::node {

bool ReadyManyAccess::ParkRegisterTimeout(
    Scheduler &scheduler, ReadyManyEntry &entry,
    const std::uint64_t timer_wait_id, const std::uint64_t stop_source_id,
    const std::uint64_t stop_generation, const std::uint64_t stop_epoch,
    const TimerDeadline &timer_deadline) noexcept {
  SchedulerState &state = *scheduler.state_;
  if (!TimerStorePush(state.ready.timers, state.ready.timer_wait_id_index,
                      TimerWait{
                          .kind = TimerWaitKind::ReactorManyTimeout,
                          .task_id = entry.record->id,
                          .wait_id = timer_wait_id,
                          .stop_source_id = stop_source_id,
                          .stop_generation = stop_generation,
                          .stop_epoch = stop_epoch,
                          .deadline = timer_deadline.deadline,
                          .deadline_ns = timer_deadline.deadline_ns,
                          .sequence = state.identity.next_timer_sequence++,
                      })) {
    return false;
  }
  ++::rund::detail::task::Stat(state.evidence.metrics,
                               ::rund::detail::task::StatSlot::Timers);
  RecordReactorTimedWaitRegistered(state.evidence.metrics);
  return true;
}

} // namespace rund::node
