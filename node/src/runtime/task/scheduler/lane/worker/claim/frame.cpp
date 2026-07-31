#include "../local.hpp"

namespace rund::node {

void LaneWorkerAccess::ResetFrame(LaneJobFrame& frame) noexcept {
  frame.id = 0u;
  frame.record = nullptr;
  frame.ticket = 0u;
  frame.job_sequence = 0u;
  frame.direct_job = false;
  frame.mailbox_job = false;
  frame.split_primitive_packets = false;
  frame.root_exclusive_commit = false;
  frame.segment_job = false;
  frame.segment_result_view_enabled = false;
  frame.root_exclusive_hot_standby = false;
  frame.ready_signal_at_job_start = 0u;
  frame.work = nullptr;
  frame.completion = {};
  frame.segment_jobs.clear();
  frame.segment_original_job_count = 0u;
  frame.segment_completed = 0u;
  frame.segment_all_completed = false;
  frame.segment_has_trap_or_failure = false;
  frame.segment_first_task_id = 0u;
  frame.segment_last_task_id = 0u;
  frame.segment_first_ticket = 0u;
  frame.segment_last_ticket = 0u;
}

}  // namespace rund::node
