#include "../backend.hpp"
#include "../backlog.hpp"
#include "../many/events.hpp"
#include "../many/store.hpp"
#include "../registration.hpp"
#include "../registry.hpp"
#include "../stats.hpp"
#include "operations.hpp"

namespace rund::node::reactor_cancel_cleanup {
namespace {

[[nodiscard]] bool RemoveRegisteredManyWaits(
    ReactorRuntime &reactor, ::rund::detail::task::StatStorage &stats,
    const std::span<const ReactorManyRequest> requests) noexcept {
  bool ok = true;
  for (const ReactorManyRequest &request : requests) {
    ReactorBacklogRemoveWait(reactor, request.wait_id);
    ReactorInterest previous_interest = ReactorInterest::None;
    if (!ReactorRegistryRemoveWait(reactor, request.wait_id, nullptr,
                                   &previous_interest)) {
      continue;
    }
    if (!ReactorRegistryCollectChangesForWaitRemove(reactor, request.fd,
                                                    previous_interest)) {
      ok = false;
    }
  }
  if (!ReactorRegistrationFlushDeferredRemoves(reactor)) {
    ok = false;
  }
  if (!ReactorBackendApplyChanges(reactor, stats).ok) {
    ok = false;
  }
  return ok;
}

} // namespace

bool CleanupGroup(Scheduler &scheduler,
                  const ReactorCleanupRequest &request) noexcept {
  ReactorManyGroup *const group = ReactorManyFindGroup(
      scheduler.state_->reactor.reactor_many_groups, request.group_id);
  if (group == nullptr) {
    return true;
  }

  const std::span<const ReactorManyRequest> requests = ReactorManyRequests(
      scheduler.state_->reactor.reactor_many_requests, *group);
  if (requests.size() != group->request_count) {
    return false;
  }

  if (request.store_event) {
    const ReactorManyRequest *const stored_request =
        ReactorManyFindRequest(requests, request.wait_id);
    if (stored_request != nullptr &&
        !ReactorManyEventSlotsAppend(
            *group, *stored_request, request.events, request.reason,
            scheduler.state_->reactor.reactor_many_event_slots)) {
      return false;
    }
  }

  TaskRecord *const record = scheduler.state_->Find(group->task_id);
  const std::uint64_t wake_wait_id =
      request.wait_id == 0u ? group->group_id : request.wait_id;
  const ReactorManyRequest *const wake_request =
      ReactorManyFindRequest(requests, request.wait_id);
  const bool can_wake = request.wake_owner && record != nullptr &&
                        record->state == TaskState::IoBlocked &&
                        record->wait_source_id == group->group_id;

  if (group->completed) {
    if (can_wake) {
      WakeTask(scheduler, *record, wake_wait_id,
               wake_request == nullptr ? kInvalidReactorHandle
                                       : wake_request->fd,
               wake_request == nullptr ? ReactorInterest::None
                                       : wake_request->interest,
               request.events, request.deadline_ns, request.reason,
               request.reason != ReasonCode::IoPollFailed);
    }
    if (request.erase_group) {
      ReactorManyEventSlotsEraseGroup(
          scheduler.state_->reactor.reactor_many_event_slots,
          group->first_request, group->request_count);
      ReactorManyEraseGroup(scheduler.state_->reactor.reactor_many_groups,
                            scheduler.state_->reactor.reactor_many_requests,
                            request.group_id);
    }
    return true;
  }

  bool cleanup_ok = true;
  if (request.cancel_timeout_timer) {
    const bool timer_expected =
        group->timer_wait_id != 0u || request.require_timeout_timer_cancel;
    const bool timer_canceled =
        group->timer_wait_id != 0u &&
        scheduler.CancelReactorTimeoutTimer(group->timer_wait_id);
    if (timer_expected && !timer_canceled) {
      cleanup_ok = false;
    }
  }
  if (request.cleanup_siblings &&
      !RemoveRegisteredManyWaits(scheduler.state_->reactor.reactor,
                                 scheduler.state_->evidence.metrics,
                                 requests)) {
    cleanup_ok = false;
  }

  group->completed = true;
  group->timed_out = cleanup_ok && request.reason == ReasonCode::IoTimedOut;
  if (request.reason == ReasonCode::TaskCancelled) {
    RecordReactorWaitsCanceled(scheduler.state_->evidence.metrics,
                               requests.size());
  }

  if (can_wake) {
    WakeTask(scheduler, *record, wake_wait_id,
             wake_request == nullptr ? kInvalidReactorHandle : wake_request->fd,
             wake_request == nullptr ? ReactorInterest::None
                                     : wake_request->interest,
             request.events, request.deadline_ns, request.reason, cleanup_ok);
  }

  if (request.erase_group || request.reason == ReasonCode::TaskCancelled) {
    ReactorManyEventSlotsEraseGroup(
        scheduler.state_->reactor.reactor_many_event_slots,
        group->first_request, group->request_count);
    ReactorManyEraseGroup(scheduler.state_->reactor.reactor_many_groups,
                          scheduler.state_->reactor.reactor_many_requests,
                          request.group_id);
  }
  return cleanup_ok;
}

} // namespace rund::node::reactor_cancel_cleanup
