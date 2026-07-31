#include "../local.hpp"

namespace rund::node {

void LaneWorkerAccess::LoadRegularJob(TaskLane& lane,
                                      LaneJobFrame& frame) noexcept {
  frame.id = lane.task_id;
  frame.record = lane.task_record;
  frame.ticket = lane.commit_ticket;
  frame.job_sequence = lane.job_sequence;
  frame.direct_job = lane.direct_job;
  frame.split_primitive_packets = lane.split_primitive_packets;
  frame.root_exclusive_commit = lane.root_exclusive_commit;
  frame.root_exclusive_hot_standby = lane.root_exclusive_hot_standby;
  lane.root_exclusive_hot_standby = false;
  lane.root_exclusive_commit = false;
  frame.ready_signal_at_job_start =
      lane.ready_signal.load(std::memory_order_acquire);
  lane.has_job = false;
  lane.running = true;
}

}  // namespace rund::node
