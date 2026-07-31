#include "local.hpp"

namespace rund::node {

[[nodiscard]] ReactorManyGroup *
FindManyGroupByTimerWaitId(std::vector<ReactorManyGroup> &groups,
                           const std::uint64_t timer_wait_id) noexcept {
  for (ReactorManyGroup &group : groups) {
    if (group.timer_wait_id == timer_wait_id) {
      return &group;
    }
  }
  return nullptr;
}

bool Scheduler::WakeReactorManyGroupFromWait(const ReactorWait &wait,
                                             const ReasonCode code,
                                             const ReactorEvent events,
                                             const bool store_event) noexcept {
  TaskRecord *const record = state_->Find(wait.task_id);
  if (record == nullptr || record->wait_source_id == 0u) {
    return false;
  }
  ReactorManyGroup *const group = ReactorManyFindGroup(
      state_->reactor.reactor_many_groups, record->wait_source_id);
  if (group == nullptr) {
    return false;
  }

  if (group->completed) {
    if (!store_event || code != ReasonCode::Ok) {
      return true;
    }
  }

  const std::span<const ReactorManyRequest> requests =
      ReactorManyRequests(state_->reactor.reactor_many_requests, *group);
  if (requests.size() != group->request_count) {
    return false;
  }

  if (store_event) {
    const ReactorManyRequest *const request =
        ReactorManyFindRequest(requests, wait.wait_id);
    if (request != nullptr && !ReactorManyEventSlotsAppend(
                                  *group, *request, events, code,
                                  state_->reactor.reactor_many_event_slots)) {
      return false;
    }
  }

  if (group->completed) {
    return true;
  }

  bool cleanup_ok = true;
  const std::uint64_t group_id = group->group_id;
  cleanup_ok = ReactorCleanupWait(
      *this, ReactorCleanupRequest{.wait_id = wait.wait_id,
                                   .group_id = group_id,
                                   .reason = code,
                                   .cancel_timeout_timer = true,
                                   .require_timeout_timer_cancel =
                                       group->timer_wait_id != 0u,
                                   .remove_ready_backlog = true,
                                   .cleanup_siblings = true,
                                   .events = events,
                                   .store_event = false});
  return cleanup_ok;
}

bool Scheduler::WakeReactorManyTimeout(const TimerWait &wait) noexcept {
  ReactorManyGroup *const group = FindManyGroupByTimerWaitId(
      state_->reactor.reactor_many_groups, wait.wait_id);
  if (group == nullptr) {
    return false;
  }
  const bool cleanup_ok = ReactorCleanupWait(
      *this, ReactorCleanupRequest{.wait_id = 0u,
                                   .group_id = group->group_id,
                                   .reason = ReasonCode::IoTimedOut,
                                   .cancel_timeout_timer = false,
                                   .remove_ready_backlog = true,
                                   .cleanup_siblings = true,
                                   .deadline_ns = wait.deadline_ns});
  if (!cleanup_ok) {
    RecordReactorTimeoutCleanupFailure(state_->evidence.metrics);
  }
  return cleanup_ok;
}

} // namespace rund::node
