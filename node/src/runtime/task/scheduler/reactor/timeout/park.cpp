#include <rund/task/stats/slots.hpp>

#include "../../../../reactor/readiness/handle.hpp"
#include "../../../../reactor/readiness/mask.hpp"
#include "../../state/model/task.hpp"
#include "../../state/storage.hpp"

#include "../../timer/store.hpp"
#include "../cleanup/request.hpp"
#include "../registry.hpp"
#include "../stats.hpp"
#include "../timeout.hpp"

namespace rund::node {

bool Scheduler::ParkTimedReactorWait(
    TaskRecord &record, const int fd, const short interest,
    const std::chrono::nanoseconds timeout,
    const std::uint64_t wait_host_handle_id, const std::uint64_t fd_generation,
    const std::uint64_t stop_source_id, const std::uint64_t stop_generation,
    const std::uint64_t stop_epoch, const ::rund::net::SocketView socket,
    std::uint64_t &wait_id, ::rund::detail::task::IoDecision &result) noexcept {
  if (!record.coroutine_task) {
    result = FailIo(ReasonCode::TaskLeafPrimitiveForbidden);
    CompletePrimitiveCommit();
    return false;
  }
  if (state_->ready.timers.size() >= state_->resources.limits.timer_capacity) {
    result = FailIo(ReasonCode::TimerCapacityExceeded);
    CompletePrimitiveCommit();
    return false;
  }
  if (ReactorRegistrySize(state_->reactor.reactor) >=
      state_->resources.limits.reactor_wait_capacity) {
    result = FailIo(ReasonCode::ReactorWaitCapacityExceeded);
    CompletePrimitiveCommit();
    return false;
  }
  if (!ReactorTimeoutReserveTimerStorage(state_->ready.timers,
                                         state_->ready.timer_wait_id_index)) {
    result = FailIo(ReasonCode::TimerCapacityExceeded);
    CompletePrimitiveCommit();
    return false;
  }

  const TimerDeadline timer_deadline = MakeTimerDeadline(timeout);
  wait_id = state_->identity.next_wait_id++;
  const ReactorWait wait{.socket = socket,
                         .task_id = record.id,
                         .wait_id = wait_id,
                         .host_handle_id = wait_host_handle_id,
                         .fd_generation = fd_generation,
                         .stop_source_id = stop_source_id,
                         .stop_generation = stop_generation,
                         .stop_epoch = stop_epoch,
                         .fd = ReactorHandleFromPublic(fd),
                         .interest = ReactorInterestFromBits(interest)};
  if (!ReactorRegistryAddWait(state_->reactor.reactor, wait)) {
    result = FailIo(ReasonCode::ReactorWaitCapacityExceeded);
    CompletePrimitiveCommit();
    return false;
  }
  if (!ReactorRegistryCollectChangesForWaitAdd(state_->reactor.reactor, wait)) {
    (void)ReactorRegistryRemoveWait(state_->reactor.reactor, wait.wait_id,
                                    nullptr);
    result = FailIo(ReasonCode::ReactorWaitCapacityExceeded);
    CompletePrimitiveCommit();
    return false;
  }

  if (!TimerStorePush(
          state_->ready.timers, state_->ready.timer_wait_id_index,
          TimerWait{.kind = TimerWaitKind::ReactorTimeout,
                    .task_id = record.id,
                    .wait_id = wait_id,
                    .stop_source_id = stop_source_id,
                    .stop_generation = stop_generation,
                    .stop_epoch = stop_epoch,
                    .deadline = timer_deadline.deadline,
                    .deadline_ns = timer_deadline.deadline_ns,
                    .sequence = state_->identity.next_timer_sequence++,
                    .host_handle_id = wait_host_handle_id,
                    .fd = fd,
                    .interest = interest})) {
    (void)ReactorCleanupWait(
        *this,
        ReactorCleanupRequest{.wait_id = wait_id,
                              .reason = ReasonCode::TimerCapacityExceeded,
                              .remove_ready_backlog = true});
    result = FailIo(ReasonCode::TimerCapacityExceeded);
    CompletePrimitiveCommit();
    return false;
  }

  ++::rund::detail::task::Stat(state_->evidence.metrics,
                               ::rund::detail::task::StatSlot::ReactorWaits);
  RecordReactorWaitRegistered(state_->evidence.metrics);
  RecordReactorTimedWaitRegistered(state_->evidence.metrics);
  ++::rund::detail::task::Stat(state_->evidence.metrics,
                               ::rund::detail::task::StatSlot::Timers);
  ++::rund::detail::task::Stat(state_->evidence.metrics,
                               ::rund::detail::task::StatSlot::Parked);
  record.state = TaskState::IoBlocked;
  record.wait_id = wait_id;
  record.io_result = ReasonCode::Ok;
  record.io_revents = 0;
  Record(::rund::detail::task::OperationKind::IoPark, ReasonCode::Ok, record.id,
         0u, wait_id, 0u, fd, interest, 0);
  Record(::rund::detail::task::OperationKind::TimerPark, ReasonCode::Ok,
         record.id, 0u, wait_id, 0u, fd, interest, 0,
         timer_deadline.deadline_ns);
  return true;
}

} // namespace rund::node
