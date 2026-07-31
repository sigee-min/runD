#include "../local.hpp"

namespace rund::node {

void LaneWorkerAccess::LoadWork(TaskLane& lane,
                                LaneJobFrame& frame) noexcept {
  frame.work = lane.work_head;
  lane.work_head = frame.work == nullptr ? nullptr : frame.work->next;
  if (lane.work_head == nullptr) {
    lane.work_tail = nullptr;
  }
  if (frame.work != nullptr) {
    frame.work->next = nullptr;
  }
  lane.running = true;
}

}  // namespace rund::node
