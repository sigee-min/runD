#include "../state/model/lane.hpp"
#include "../state/model/work.hpp"
#include "../state/storage.hpp"

namespace rund::node {

bool Scheduler::EnqueueWork(SchedulerWork* const work,
                            const std::uint32_t worker) noexcept {
  if (work == nullptr || work->invoke == nullptr || state_ == nullptr ||
      worker >= state_->lanes.lanes.size()) {
    return false;
  }
  TaskLane& lane = *state_->lanes.lanes[worker];
  bool notify = false;
  {
    std::lock_guard lock{lane.mutex};
    if (lane.stop || work->next != nullptr) {
      return false;
    }
    if (lane.work_tail == nullptr) {
      lane.work_head = work;
    } else {
      lane.work_tail->next = work;
    }
    lane.work_tail = work;
    notify = lane.ready_wait_requested;
  }
  if (notify) {
    lane.ready.notify_one();
  }
  return true;
}

}  // namespace rund::node
