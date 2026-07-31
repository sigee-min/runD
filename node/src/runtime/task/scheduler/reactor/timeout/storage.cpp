#include "../timeout.hpp"

#include "../../state/storage.hpp"
#include "../../timer/store.hpp"
#include "../stats.hpp"

namespace rund::node {

bool ReactorTimeoutReserveTimerStorage(
    std::vector<TimerWait> &timers,
    std::vector<TimerWaitIdIndexEntry> &index) noexcept {
  try {
    timers.reserve(timers.size() + 1u);
    index.reserve(index.size() + 1u);
  } catch (...) {
    return false;
  }
  return true;
}

bool ReactorTimeoutCancelTimer(std::vector<TimerWait> &timers,
                               std::vector<TimerWaitIdIndexEntry> &index,
                               const std::uint64_t wait_id) noexcept {
  const TimerWait *const found = TimerStoreFind(timers, index, wait_id);
  if (found == nullptr || found->kind == TimerWaitKind::Sleep) {
    return false;
  }
  return TimerStoreCancel(timers, index, wait_id);
}

bool Scheduler::CancelReactorTimeoutTimer(
    const std::uint64_t wait_id) noexcept {
  const bool canceled = ReactorTimeoutCancelTimer(
      state_->ready.timers, state_->ready.timer_wait_id_index, wait_id);
  if (canceled) {
    RecordReactorTimeoutTimerCancel(state_->evidence.metrics);
  }
  return canceled;
}

} // namespace rund::node
