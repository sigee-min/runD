#include "../local.hpp"

namespace rund::node {

void LaneWorkerAccess::LoadMailboxJob(TaskLane& lane,
                                      LaneJobFrame& frame) noexcept {
  frame.id = lane.mailbox_task_id.load(std::memory_order_relaxed);
  frame.record = lane.mailbox_record.load(std::memory_order_relaxed);
  frame.ticket = lane.mailbox_commit_ticket.load(std::memory_order_relaxed);
  frame.job_sequence =
      lane.mailbox_job_sequence.load(std::memory_order_relaxed);
  frame.direct_job = true;
  frame.mailbox_job = true;
  frame.split_primitive_packets = true;
  frame.ready_signal_at_job_start =
      lane.ready_signal.load(std::memory_order_acquire);
}

}  // namespace rund::node
