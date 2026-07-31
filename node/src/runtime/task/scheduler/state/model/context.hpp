#pragma once

#include <rund/reason.hpp>
#include <rund/task/operation/kind.hpp>

#include <cstdint>

namespace rund::node {

struct LaneSegmentEffect;

struct SchedulerThreadContext {
  void *scheduler = nullptr;
  std::uint64_t task_id = 0u;
  std::uint64_t scope_id = 1u;
  std::uint64_t commit_ticket = 0u;
  bool commit_acquired = false;
  bool root_submit_recorded = false;
  bool pending_root_submit = false;
  bool root_exclusive_commit = false;
  bool split_primitive_packets = false;
  bool lane_owned_segment_active = false;
  bool lane_owned_segment_trapped = false;
  bool lane_executor_active = false;
  void *lane_executor_lane = nullptr;
  LaneSegmentEffect *lane_effect = nullptr;
  std::uint64_t deferred_hot_path_ensure_skips = 0u;
  void *record = nullptr;
  SchedulerThreadContext *previous = nullptr;
};

extern thread_local SchedulerThreadContext *active_scheduler_context;

} // namespace rund::node
