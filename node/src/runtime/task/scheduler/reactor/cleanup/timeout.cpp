#include "../../timer/store.hpp"
#include "operations.hpp"

namespace rund::node::reactor_cancel_cleanup {

bool CancelTimeoutTimer(Scheduler &scheduler, const std::uint64_t wait_id,
                        const ReactorTimeoutCleanupPolicy policy) noexcept {
  switch (policy) {
  case ReactorTimeoutCleanupPolicy::None:
    return true;
  case ReactorTimeoutCleanupPolicy::IfPresent:
    if (wait_id == 0u ||
        !TimerStoreContains(scheduler.state_->ready.timers,
                            scheduler.state_->ready.timer_wait_id_index,
                            wait_id)) {
      return true;
    }
    return scheduler.CancelReactorTimeoutTimer(wait_id);
  case ReactorTimeoutCleanupPolicy::Required:
    return wait_id != 0u && scheduler.CancelReactorTimeoutTimer(wait_id);
  }
  return false;
}

}  // namespace rund::node::reactor_cancel_cleanup
