#include "../../state/model/task.hpp"
#include "../../state/segment.hpp"
#include "../../state/storage.hpp"

namespace rund::node {

void Scheduler::RestoreLaneOwnedSegmentTasks(
    const std::vector<std::uint64_t> &task_ids) noexcept {
  for (auto it = task_ids.rbegin(); it != task_ids.rend(); ++it) {
    (void)RequeueReadyTask(*it, 0u);
  }
}

bool Scheduler::PlanLaneOwnedSegments(
    const std::vector<std::uint64_t> &task_ids,
    std::vector<LaneOwnedSegmentLane> &segments,
    std::size_t *const planned_logical_tasks) noexcept {
  *planned_logical_tasks = 0u;
  const std::size_t available_lanes = state_->lanes.lanes.size();
  try {
    segments.resize(available_lanes);
    for (LaneOwnedSegmentLane &segment : segments) {
      segment.reset();
    }
    state_->lanes.segment_commit_lanes.clear();
    state_->lanes.segment_commit_lanes.reserve(task_ids.size());
  } catch (...) {
    return false;
  }
  std::size_t canonical_index = 0u;
  for (std::size_t index = 0u; index < task_ids.size(); ++index) {
    const std::uint64_t id = task_ids[index];
    TaskRecord *const record = state_->Find(id);
    if (record == nullptr || record->state != TaskState::Ready ||
        record->lane_segment_side_exit) {
      continue;
    }
    if (record->coroutine_task) {
      record->lane_segment_side_exit = true;
      continue;
    }
    std::size_t lane_index = state_->LaneIndexForTask(record, available_lanes);
    if (lane_index >= segments.size()) {
      continue;
    }
    try {
      segments[lane_index].jobs.push_back(LaneSegmentJob{
          .task_id = id,
          .record = record,
          .logical_ticket = 0u,
          .canonical_index = canonical_index,
          .split_primitive_packets = true,
      });
      state_->lanes.segment_commit_lanes.push_back(
          static_cast<std::uint32_t>(lane_index));
      ++canonical_index;
    } catch (...) {
      return false;
    }
  }
  try {
    for (LaneOwnedSegmentLane &segment : segments) {
      segment.effects.reserve(segment.jobs.size());
    }
  } catch (...) {
    return false;
  }
  *planned_logical_tasks = canonical_index;
  return true;
}

} // namespace rund::node
