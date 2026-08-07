#include "../backend.hpp"
#include "../backlog.hpp"
#include "../registration.hpp"
#include "../registry.hpp"
#include "../stats.hpp"
#include "operations.hpp"

namespace rund::node::reactor_cancel_cleanup {

bool CleanupSingleWait(Scheduler &scheduler,
                       ReactorCleanupRequest request) noexcept {
  ReactorRuntime &reactor = scheduler.state_->reactor.reactor;
  if (request.remove_ready_backlog) {
    ReactorBacklogRemoveWait(reactor, request.wait_id);
  }

  ReactorWait removed{};
  ReactorInterest interest_before_remove = ReactorInterest::None;
  const bool wait_present = ReactorRegistryRemoveWait(
      reactor, request.wait_id, &removed, &interest_before_remove);
  if (!wait_present) {
    return true;
  }

  const std::uint64_t group_id =
      [record = scheduler.state_->Find(removed.task_id)]() noexcept {
        return record == nullptr ? 0u : record->wait_source_id;
      }();
  if (request.cleanup_siblings && group_id != 0u) {
    request.group_id = group_id;
    return CleanupGroup(scheduler, request);
  }

  bool cleanup_ok = true;
  if (!ReactorRegistryCollectChangesForWaitRemove(reactor, removed.fd,
                                                  interest_before_remove)) {
    cleanup_ok = false;
  }
  if (!ReactorRegistrationFlushDeferredRemoves(reactor)) {
    cleanup_ok = false;
  }
  const ReactorApplyResult applied =
      ReactorBackendApplyChanges(reactor, scheduler.state_->evidence.metrics);
  if (applied.disposition() != ReactorApplyDisposition::Success) {
    cleanup_ok = false;
  }
  if (request.cancel_timeout_timer &&
      !CancelTimeoutTimer(scheduler, removed.wait_id,
                          request.require_timeout_timer_cancel)) {
    cleanup_ok = false;
  }
  if (request.reason == ReasonCode::TaskCancelled) {
    RecordReactorWaitsCanceled(scheduler.state_->evidence.metrics, 1u);
  }

  TaskRecord *const record = scheduler.state_->Find(removed.task_id);
  if (!request.wake_owner || record == nullptr ||
      record->state != TaskState::IoBlocked ||
      record->wait_id != removed.wait_id) {
    return cleanup_ok;
  }
  WakeTask(scheduler, *record, removed.wait_id, removed.fd, removed.interest,
           request.events, request.deadline_ns, request.reason, cleanup_ok);
  return cleanup_ok;
}

} // namespace rund::node::reactor_cancel_cleanup
