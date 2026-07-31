#include "worker/local.hpp"

#include "../state/model/batch.hpp"

#include <algorithm>

namespace rund::node {

SchedulerLaneState::SchedulerLaneState() noexcept = default;
SchedulerLaneState::~SchedulerLaneState() = default;

bool Scheduler::StartLanes() noexcept {
  StopLanes();
  const std::uint32_t lane_count =
      std::max<std::uint32_t>(1u, state_->resources.limits.task_workers);
  try {
    state_->lanes.lanes.clear();
    state_->lanes.lane_participated.assign(lane_count, 0u);
    const std::size_t ready_capacity =
        state_->resources.limits.ready_queue_capacity;
    state_->ready.ready_depth = state_->ready.ready.size();
    state_->lanes.lane_batch_used_scratch.assign(lane_count, 0u);
    state_->lanes.ready_batch_scratch.clear();
    state_->lanes.ready_batch_scratch.reserve(ready_capacity);
    state_->lanes.ready_deferred.clear();
    state_->lanes.ready_deferred.reserve(
        state_->resources.limits.task_capacity);
    state_->lanes.submitted_batch_scratch.clear();
    state_->lanes.submitted_batch_scratch.reserve(ready_capacity);
    state_->lanes.segment_commit_lanes.clear();
    state_->lanes.segment_commit_lanes.reserve(
        state_->resources.limits.task_capacity);
    state_->lanes.lanes.reserve(lane_count);
    for (std::uint32_t index = 0u; index < lane_count; ++index) {
      state_->lanes.lanes.push_back(std::make_unique<TaskLane>());
    }
    state_->lanes.lane_threads.reserve(lane_count);
    for (std::uint32_t index = 0u; index < lane_count; ++index) {
      TaskLane* const lane = state_->lanes.lanes[index].get();
      state_->lanes.lane_threads.emplace_back([this, lane] {
        Scheduler::SetActive(this);
        active_task_lane = lane;
        rund::kernel::executor_detail::ScopedParallelRuntimeProvider
            provider_scope{state_->lanes.kernel_provider};
        LaneJobFrame frame{};
        while (ClaimLaneJob(*lane, frame)) {
          RunLaneJob(*this, *lane, frame);
          PublishLaneJob(*this, *lane, frame);
        }
        active_task_lane = nullptr;
        Scheduler::SetActive(nullptr);
      });
    }
  } catch (...) {
    StopLanes();
    state_->lanes.lanes_started = false;
    state_->lanes.lane_code = ReasonCode::TaskWorkersInvalid;
    return false;
  }
  state_->lanes.lanes_started = state_->lanes.lane_threads.size() == lane_count;
  state_->lanes.lane_code = state_->lanes.lanes_started
                                ? ReasonCode::Ok
                                : ReasonCode::TaskWorkersInvalid;
  return state_->lanes.lanes_started;
}

void Scheduler::StopLanes() noexcept {
  for (const auto& lane_ptr : state_->lanes.lanes) {
    if (lane_ptr) {
      {
        std::lock_guard<std::mutex> lock(lane_ptr->mutex);
        lane_ptr->stop = true;
      }
      lane_ptr->ready.notify_one();
    }
  }
  for (std::thread& thread : state_->lanes.lane_threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  state_->lanes.lane_threads.clear();
  state_->lanes.lanes.clear();
  state_->lanes.lane_participated.clear();
  state_->lanes.segment_commit_lanes.clear();
  state_->ready.ready_depth = state_->ready.ready.size();
  state_->lanes.lanes_started = false;
}

bool Scheduler::Ready() const noexcept {
  return state_->lanes.lanes_started && !state_->lanes.lanes.empty();
}

const char* Scheduler::Reason() const noexcept {
  return ReasonString(Code());
}

ReasonCode Scheduler::Code() const noexcept { return state_->lanes.lane_code; }

}  // namespace rund::node
