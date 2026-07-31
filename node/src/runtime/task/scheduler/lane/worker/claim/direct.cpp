#include "../local.hpp"

namespace rund::node {

void LaneWorkerAccess::LoadDirectReadyJob(TaskLane& lane,
                                          LaneJobFrame& frame) noexcept {
  TaskRecord* const record = lane.direct_ready_head;
  lane.direct_ready_head = record == nullptr ? nullptr : record->wake_next;
  if (lane.direct_ready_head == nullptr) {
    lane.direct_ready_tail = nullptr;
  }
  if (record == nullptr) {
    return;
  }
  frame.id = record->id;
  frame.record = record;
  frame.ticket = record->wake_ticket;
  frame.job_sequence = record->wait_token;
  frame.direct_job = true;
  frame.split_primitive_packets = true;
  frame.ready_signal_at_job_start =
      lane.ready_signal.load(std::memory_order_acquire);
  record->wake_next = nullptr;
  record->wake_ticket = 0u;
  record->wait_token = 0u;
  lane.running = true;
}

} // namespace rund::node
