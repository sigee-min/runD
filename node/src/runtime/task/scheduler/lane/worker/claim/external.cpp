#include "../local.hpp"

namespace rund::node {

void LaneWorkerAccess::LoadExternalWakeJob(TaskLane &lane,
                                           LaneJobFrame &frame) noexcept {
  TaskRecord *const record = lane.external_wake_head;
  lane.external_wake_head = record == nullptr ? nullptr : record->wake_next;
  if (lane.external_wake_head == nullptr) {
    lane.external_wake_tail = nullptr;
  }
  if (record == nullptr) {
    return;
  }
  record->wake_next = nullptr;
  frame.id = record->id;
  frame.record = record;
  frame.ticket = record->wake_ticket;
  frame.job_sequence = lane.next_job_sequence++;
  frame.direct_job = true;
  frame.split_primitive_packets = true;
  frame.ready_signal_at_job_start =
      lane.ready_signal.load(std::memory_order_acquire);
  lane.running = true;
}

} // namespace rund::node
