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
                                   .timeout_cleanup =
                                       ReactorTimeoutCleanupPolicy::None,
                                   .remove_ready_backlog = true,
                                   .cleanup_siblings = true,
                                   .deadline_ns = wait.deadline_ns});
  if (!cleanup_ok) {
    RecordReactorTimeoutCleanupFailure(state_->evidence.metrics);
  }
  return cleanup_ok;
}

} // namespace rund::node
