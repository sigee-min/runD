#include "../../state/model/lane.hpp"
#include "../../state/model/task.hpp"
#include "../../state/model/wake.hpp"
#include "../../state/storage.hpp"


namespace rund::node {

bool Scheduler::EnqueueExternalWake(const ExternalWake wake) noexcept {
  if (state_->lanes.lanes.empty() || wake.record == nullptr) {
    return false;
  }
  const std::size_t lane_index = wake.lane % state_->lanes.lanes.size();
  if (lane_index >= state_->lanes.lanes.size()) {
    return false;
  }
  TaskLane &lane = *state_->lanes.lanes[lane_index];
  bool notify = false;
  {
    std::lock_guard lock{lane.mutex};
    TaskRecord& record = *wake.record;
    if (lane.stop || record.id != wake.id ||
        record.state != TaskState::ExternalBlocked ||
        record.wake_ticket != 0u) {
      return false;
    }
    record.wake_ticket = IssueCommitTicket();
    record.wake_next = nullptr;
    if (lane.external_wake_tail == nullptr) {
      lane.external_wake_head = &record;
    } else {
      lane.external_wake_tail->wake_next = &record;
    }
    lane.external_wake_tail = &record;
    notify = lane.ready_wait_requested;
  }
  if (notify) {
    lane.ready.notify_one();
  }
  return true;
}

bool Scheduler::WakeExternal(const ExternalWake wake) noexcept {
  return EnqueueExternalWake(wake);
}

} // namespace rund::node
