#include "../../timer/store.hpp"
#include "operations.hpp"

namespace rund::node::reactor_cancel_cleanup {

bool CancelTimeoutTimer(Scheduler& scheduler, const std::uint64_t wait_id,
                        const bool require_timeout_timer_cancel) noexcept {
  const bool timer_present =
      TimerStoreContains(scheduler.state_->ready.timers,
                         scheduler.state_->ready.timer_wait_id_index, wait_id);
  const bool timer_canceled = scheduler.CancelReactorTimeoutTimer(wait_id);
  return !((timer_present || require_timeout_timer_cancel) && !timer_canceled);
}

}  // namespace rund::node::reactor_cancel_cleanup
