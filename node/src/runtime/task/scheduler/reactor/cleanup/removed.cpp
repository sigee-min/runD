#include "../backlog.hpp"
#include "../stats.hpp"
#include "operations.hpp"

namespace rund::node {

bool ReactorCleanupRemovedWait(
    Scheduler &scheduler,
    const ReactorRemovedWaitCleanupRequest request) noexcept {
  if (request.remove_ready_backlog) {
    ReactorBacklogRemoveWait(scheduler.state_->reactor.reactor,
                             request.wait.wait_id);
  }

  TaskRecord *const record = scheduler.state_->Find(request.wait.task_id);
  const std::uint64_t group_id =
      record == nullptr ? 0u : record->wait_source_id;
  if (group_id != 0u) {
    return ReactorCleanupWait(
        scheduler, ReactorCleanupRequest{
                       .wait_id = request.wait.wait_id,
                       .group_id = group_id,
                       .reason = request.reason,
                       .cancel_timeout_timer = request.cancel_timeout_timer,
                       .require_timeout_timer_cancel =
                           request.require_timeout_timer_cancel,
                       .remove_ready_backlog = request.remove_ready_backlog,
                       .cleanup_siblings = request.cleanup_siblings,
                       .erase_group = false,
                       .wake_owner = request.wake_owner,
                       .events = request.events,
                       .store_event = request.store_event,
                       .deadline_ns = request.deadline_ns});
  }

  bool cleanup_ok = true;
  if (request.cancel_timeout_timer &&
      !reactor_cancel_cleanup::CancelTimeoutTimer(
          scheduler, request.wait.wait_id,
          request.require_timeout_timer_cancel)) {
    cleanup_ok = false;
  }
  if (request.reason == ReasonCode::TaskCancelled) {
    RecordReactorWaitsCanceled(scheduler.state_->evidence.metrics, 1u);
  }
  if (!request.wake_owner || record == nullptr ||
      record->state != TaskState::IoBlocked ||
      record->wait_id != request.wait.wait_id) {
    return cleanup_ok;
  }
  reactor_cancel_cleanup::WakeTask(scheduler, *record, request.wait.wait_id,
                                   request.wait.fd, request.wait.interest,
                                   request.events, request.deadline_ns,
                                   request.reason, cleanup_ok);
  return cleanup_ok;
}

} // namespace rund::node
