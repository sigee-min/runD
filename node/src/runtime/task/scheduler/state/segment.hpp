#pragma once

#include "../hash.hpp"
#include "model/segment.hpp"

#include <vector>

namespace rund::node {

struct LaneOwnedSegmentLane {
  std::vector<LaneSegmentJob> jobs{};
  std::vector<LaneSegmentEffect> effects{};
  std::uint64_t first_ticket = 0u;
  std::uint64_t last_ticket = 0u;
  std::uint64_t first_task_id = 0u;
  std::uint64_t last_task_id = 0u;
  std::uint64_t sequence = 0u;
  std::size_t completed = 0u;
  std::size_t effect_count = 0u;
  bool all_completed = false;
  bool has_trap_or_failure = false;
  bool submitted = false;
  bool notify_ready = false;
  bool used_result_view = false;

  void reset() noexcept {
    jobs.clear();
    effects.clear();
    first_ticket = 0u;
    last_ticket = 0u;
    first_task_id = 0u;
    last_task_id = 0u;
    sequence = 0u;
    completed = 0u;
    effect_count = 0u;
    all_completed = false;
    has_trap_or_failure = false;
    submitted = false;
    notify_ready = false;
    used_result_view = false;
  }
};

struct LaneOwnedSegmentSummary {
  bool success = true;
  std::uint64_t first_ticket = 0u;
  std::uint64_t last_ticket = 0u;
  std::uint64_t submitted_lanes = 0u;
  std::uint64_t commit_logical_events = 0u;
};

struct LaneOwnedTerminalRange {
  ::rund::detail::task::OperationKind kind =
      ::rund::detail::task::OperationKind::None;
  ReasonCode code = ReasonCode::Ok;
  std::uint64_t first_task_id = 0u;
  std::uint64_t last_task_id = 0u;
  std::uint64_t first_ticket = 0u;
  std::uint64_t last_ticket = 0u;
  std::uint64_t logical_tasks = 0u;
  std::uint64_t order_hash = kFnvOffset;
  bool active = false;
};

} // namespace rund::node
