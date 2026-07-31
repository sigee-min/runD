#include "../local.hpp"

namespace rund::node {

void LaneWorkerAccess::LoadSegmentJob(TaskLane& lane,
                                      LaneJobFrame& frame) noexcept {
  frame.segment_jobs.swap(lane.segment_jobs);
  frame.job_sequence = lane.job_sequence;
  frame.segment_result_view_enabled = lane.segment_result_view_enabled;
  lane.has_job = false;
  lane.running = true;
  frame.segment_job = true;
}

}  // namespace rund::node
