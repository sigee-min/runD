#include "../../state/model/context.hpp"
#include "../../state/segment.hpp"
#include "../../state/storage.hpp"
#include "../residual.hpp"
#include "../residual/policy.hpp"

namespace rund::node {

bool Scheduler::RunLaneOwnedSegments(
    const std::vector<std::uint64_t> &task_ids) noexcept {
  if (task_ids.empty()) {
    return false;
  }
  if (state_->lanes.lanes.empty() || active_scheduler_context != nullptr) {
    return RunReadyBatch(task_ids);
  }
  if (task_ids.size() == 1u) {
    return RunReadyBatch(task_ids);
  }

  std::vector<LaneOwnedSegmentLane> &segments = lane_segments_;
  std::size_t planned_logical_tasks = 0u;
  if (!PlanLaneOwnedSegments(task_ids, segments, &planned_logical_tasks)) {
    return RunReadyBatch(task_ids);
  }
  if (planned_logical_tasks == 0u) {
    RestoreLaneOwnedSegmentTasks(task_ids);
    return true;
  }
  std::vector<std::uint64_t> &executed_task_ids = lane_executed_;
  std::vector<LaneSegmentEffect> &lane_effects = lane_effects_;
  try {
    executed_task_ids.clear();
    lane_effects.clear();
    executed_task_ids.reserve(planned_logical_tasks);
    lane_effects.reserve(planned_logical_tasks);
  } catch (...) {
    return RunReadyBatch(task_ids);
  }
  const std::size_t submitted_lanes =
      PublishLaneOwnedSegments(segments, planned_logical_tasks);
  if (submitted_lanes == 0u) {
    return RunReadyBatch(task_ids);
  }
  const std::size_t logical_tasks = planned_logical_tasks;
  RecordLaneOwnedSegmentAdmission(submitted_lanes, logical_tasks);
  NotifyLaneOwnedSegments(segments, submitted_lanes);

  LaneOwnedSegmentSummary summary{};
  DrainLaneOwnedSegmentResults(segments, &summary);
  std::lock_guard evidence_lock{state_->evidence.mutex};
  const auto record_lane_residual_policy =
      [this, submitted_lanes, logical_tasks, &task_ids](
          const bool all_success_terminal, const bool has_side_exit) noexcept {
        if (submitted_lanes <= 1u) {
          state_->batches.lane_residual_join_owner_policy =
              SchedulerBatchState::LaneResidualJoinOwnerPolicy{};
          return;
        }
        const LaneResidualPolicyDecision lane_residual_policy =
            DecideLaneResidualPolicy(LaneResidualPolicyInput{
                .participating_lanes =
                    static_cast<std::uint32_t>(submitted_lanes),
                .logical_tasks =
                    static_cast<std::uint32_t>(std::min<std::size_t>(
                        logical_tasks,
                        std::numeric_limits<std::uint32_t>::max())),
                .all_success_terminal = all_success_terminal,
                .canonical_home_lanes_preserved = true,
                .has_side_exit = has_side_exit,
            });
        RecordLaneResidualPolicyDecision(state_->evidence.metrics,
                                         lane_residual_policy.accepted);
        if (lane_residual_policy.accepted && !task_ids.empty()) {
          state_->batches.lane_residual_join_owner_policy.active = true;
          state_->batches.lane_residual_join_owner_policy.first_task_id =
              task_ids.front();
          state_->batches.lane_residual_join_owner_policy.last_task_id =
              task_ids.back();
        } else {
          state_->batches.lane_residual_join_owner_policy =
              SchedulerBatchState::LaneResidualJoinOwnerPolicy{};
        }
      };
  if (CommitSuccessfulLaneOwnedSegments(task_ids, logical_tasks, summary)) {
    record_lane_residual_policy(true, false);
    return true;
  }

  CollectLaneOwnedSegmentEffects(segments, executed_task_ids, lane_effects);
  CommitLaneOwnedSegmentEffects(lane_effects);
  const auto lane_effects_resolve_terminal_success = [&]() noexcept {
    if (logical_tasks == 0u || task_ids.empty() ||
        executed_task_ids.size() != logical_tasks) {
      return false;
    }
    if (task_ids.front() == 0u || task_ids.back() < task_ids.front() ||
        task_ids.back() - task_ids.front() + 1u != logical_tasks) {
      return false;
    }
    for (std::size_t offset = 0u; offset < executed_task_ids.size(); ++offset) {
      if (executed_task_ids[offset] != task_ids.front() + offset) {
        return false;
      }
    }
    std::size_t terminal_success_count = 0u;
    for (const LaneSegmentEffect &effect : lane_effects) {
      if (effect.trapped ||
          (effect.terminal &&
           (effect.terminal_kind !=
                ::rund::detail::task::OperationKind::Complete ||
            effect.code != ReasonCode::Ok))) {
        return false;
      }
      if (effect.terminal &&
          effect.terminal_kind ==
              ::rund::detail::task::OperationKind::Complete &&
          effect.code == ReasonCode::Ok) {
        ++terminal_success_count;
      }
    }
    return terminal_success_count == logical_tasks;
  };
  const bool lane_effects_terminal_success =
      lane_effects_resolve_terminal_success();
  record_lane_residual_policy(lane_effects_terminal_success,
                              !lane_effects_terminal_success);
  MarkLaneOwnedSegmentSideExits(task_ids, executed_task_ids);
  RequeueLaneOwnedSegmentSideExits(task_ids);
  return true;
}

} // namespace rund::node
