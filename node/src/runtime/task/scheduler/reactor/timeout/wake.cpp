#include "../../state/model/timer.hpp"
#include "../../state/storage.hpp"

#include "../cleanup/request.hpp"
#include "../stats.hpp"

namespace rund::node {

bool Scheduler::WakeReactorTimeout(const TimerWait &wait) noexcept {
  const bool cleanup_ok = ReactorCleanupWait(
      *this, ReactorCleanupRequest{.wait_id = wait.wait_id,
                                   .group_id = 0u,
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
