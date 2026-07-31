#include <rund/task/stats/slots.hpp>

#include "../../state/model/batch.hpp"
#include "../../state/model/task.hpp"
#include "../../state/storage.hpp"
#include "../../reactor/registry.hpp"

#include <algorithm>

namespace rund::node {

bool Scheduler::RunMultiLaneReadyBatch(const std::uint64_t id) noexcept {
  try {
    if (state_->lanes.ready_batch_scratch.capacity() >=
            state_->lanes.lanes.size() &&
        state_->lanes.ready_deferred.capacity() >=
            state_->lanes.lanes.size() &&
        state_->lanes.lane_batch_used_scratch.capacity() >=
            state_->lanes.lanes.size() &&
        state_->lanes.submitted_batch_scratch.capacity() >=
            state_->resources.limits.ready_queue_capacity) {
      ++::rund::detail::task::Stat(
          state_->evidence.metrics,
          ::rund::detail::task::StatSlot::LaneDispatchBatchScratchReuses);
    }
    state_->lanes.ready_batch_scratch.clear();
    state_->lanes.ready_deferred.clear();
    state_->lanes.lane_batch_used_scratch.assign(state_->lanes.lanes.size(),
                                                 0u);
    state_->lanes.ready_batch_scratch.reserve(state_->lanes.lanes.size());
    state_->lanes.ready_deferred.reserve(state_->lanes.lanes.size());
  } catch (...) {
    return RunOnLane(id);
  }
  std::vector<std::uint64_t> &batch = state_->lanes.ready_batch_scratch;
  std::vector<std::uint64_t> &skipped = state_->lanes.ready_deferred;
  std::vector<std::uint8_t> &used_lanes = state_->lanes.lane_batch_used_scratch;
  const bool lane_owned_eligible = state_->ready.timers.empty() &&
                                   ReactorRegistryEmpty(
                                       state_->reactor.reactor) &&
                                   state_->batches.direct_jobs_in_flight == 0u;
  bool duplicate_lane_batch = false;
  batch.push_back(id);
  used_lanes[state_->LaneIndexForTask(id, state_->lanes.lanes.size())] = 1u;
  const std::size_t segment_limit =
      state_->resources.limits.ready_queue_capacity;
  while (batch.size() < segment_limit) {
    const std::uint64_t candidate = PopReady(0u);
    if (candidate == 0u) {
      break;
    }
    TaskRecord *const record = state_->Find(candidate);
    if (record == nullptr || record->state != TaskState::Ready) {
      continue;
    }
    if (record->lane_segment_side_exit) {
      skipped.push_back(candidate);
      break;
    }
    const std::size_t lane_index =
        state_->LaneIndexForTask(record, state_->lanes.lanes.size());
    if (used_lanes[lane_index] == 1u) {
      if (lane_owned_eligible) {
        duplicate_lane_batch = true;
        batch.push_back(candidate);
      } else {
        skipped.push_back(candidate);
        break;
      }
      continue;
    }
    if (!CanSubmitToLane(candidate)) {
      skipped.push_back(candidate);
      break;
    }
    used_lanes[lane_index] = 1u;
    batch.push_back(candidate);
  }
  for (auto skipped_it = skipped.rbegin(); skipped_it != skipped.rend();
       ++skipped_it) {
    RestoreReadyFront(*skipped_it, 0u);
  }
  if (lane_owned_eligible &&
      (batch.size() > state_->lanes.lanes.size() || duplicate_lane_batch)) {
    return RunLaneOwnedSegments(batch);
  }
  return RunReadyBatch(batch);
}

} // namespace rund::node
