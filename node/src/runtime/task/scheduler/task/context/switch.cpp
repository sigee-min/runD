#include "local.hpp"

#include <cstdint>

namespace rund::node {

void Scheduler::RunTaskQuantum(TaskRecord& record,
                                const std::uint64_t commit_ticket,
                                const bool split_primitive_packets,
                                const bool lane_owned_segment,
                                const bool root_exclusive_commit,
                                CompletionLease *const completion) noexcept {
  {
    std::lock_guard evidence_lock{state_->evidence.mutex};
    record.quantum_active = true;
    if (!lane_owned_segment) {
      record.lane_segment_side_exit = false;
    }
  }
  if (record.coroutine_task) {
    RunCoroutineQuantum(record, commit_ticket, split_primitive_packets,
                        lane_owned_segment, root_exclusive_commit, completion);
    return;
  }
  RunLeafQuantum(record, commit_ticket, split_primitive_packets,
                 lane_owned_segment, root_exclusive_commit);
}

}  // namespace rund::node
