#include <rund/task/stats/slots.hpp>

#include "../../state/model/lane.hpp"
#include "../../state/segment.hpp"
#include "../../state/storage.hpp"
#include "../residual.hpp"

namespace rund::node {

namespace {

[[nodiscard]] bool
CanUseLaneSegmentResultView(const LaneSegmentResultView &view,
                            const auto &segment) noexcept {
  return view.all_completed && view.completed == segment.jobs.size() &&
         !view.has_trap_or_failure;
}

void ConsumeLaneSegmentResultView(const LaneSegmentResultView &view, auto &lane,
                                  auto &segment) noexcept {
  segment.completed = view.completed;
  segment.first_task_id = lane.segment_first_task_id;
  segment.last_task_id = lane.segment_last_task_id;
  segment.first_ticket = lane.segment_first_ticket;
  segment.last_ticket = lane.segment_last_ticket;
  segment.all_completed = lane.segment_all_completed;
  segment.has_trap_or_failure = lane.segment_has_trap_or_failure;
  segment.used_result_view = true;
  segment.effects.swap(lane.segment_effects);
  segment.effect_count = segment.effects.size();
}

} // namespace

void Scheduler::DrainLaneOwnedSegmentResults(
    std::vector<LaneOwnedSegmentLane> &segments,
    LaneOwnedSegmentSummary *const summary) noexcept {
  summary->success = true;
  summary->first_ticket = 0u;
  summary->last_ticket = 0u;
  summary->submitted_lanes = 0u;
  summary->commit_logical_events = 0u;
  LaneCompletionGroup completion_group{};
  std::size_t submitted_lane_count = 0u;
  for (const LaneOwnedSegmentLane &segment : segments) {
    if (segment.submitted) {
      ++submitted_lane_count;
    }
  }
  const bool use_completion_group = submitted_lane_count > 1u;
  const bool record_lane_residual =
      ShouldRecordLaneResidualSegmentMetrics(submitted_lane_count);
  for (std::size_t lane_index = 0u; lane_index < segments.size();
       ++lane_index) {
    LaneOwnedSegmentLane &segment = segments[lane_index];
    if (!segment.submitted) {
      continue;
    }
    if (use_completion_group && !completion_group.active) {
      completion_group.first_sequence = segment.sequence;
      completion_group.active = true;
      ++::rund::detail::task::Stat(
          state_->evidence.metrics,
          ::rund::detail::task::StatSlot::LaneCompletionGroupAdmissions);
    }
    if (use_completion_group) {
      ++completion_group.expected_lanes;
    }
    ++summary->submitted_lanes;
    TaskLane &lane = *state_->lanes.lanes[lane_index];
    const bool group_completion_observed =
        use_completion_group &&
        TryConsumeLaneCompletionGroupSignal(lane, segment.sequence);
    if (group_completion_observed) {
      ++::rund::detail::task::Stat(
          state_->evidence.metrics,
          ::rund::detail::task::StatSlot::LaneLocalCompletionPackets);
    } else if (!use_completion_group &&
               TryConsumeCompletedJobSpin(lane, segment.sequence)) {
      ++::rund::detail::task::Stat(
          state_->evidence.metrics,
          ::rund::detail::task::StatSlot::LaneLocalCompletionPackets);
    } else {
      (void)WaitForLaneCompletionSignal(lane, segment.sequence);
      ++::rund::detail::task::Stat(
          state_->evidence.metrics,
          ::rund::detail::task::StatSlot::LaneLocalCompletionPackets);
    }
    if (use_completion_group) {
      ++completion_group.observed_lanes;
    }
    segment.jobs.swap(lane.segment_jobs);
    if (record_lane_residual) {
      RecordLaneResidualCompletionPublish(state_->evidence.metrics);
    }
    const bool view_sequence_matches =
        use_completion_group &&
        lane.segment_result_sequence.load(std::memory_order_acquire) ==
            segment.sequence;
    if (view_sequence_matches &&
        CanUseLaneSegmentResultView(lane.segment_result_view, segment)) {
      ConsumeLaneSegmentResultView(lane.segment_result_view, lane, segment);
      ++::rund::detail::task::Stat(
          state_->evidence.metrics,
          ::rund::detail::task::StatSlot::LaneCompletionMutexLocksAvoided);
      lane.root_reserved.store(false, std::memory_order_release);
    } else {
      if (use_completion_group) {
        ++::rund::detail::task::Stat(
            state_->evidence.metrics,
            ::rund::detail::task::StatSlot::LaneCompletionGroupRejections);
      }
      {
        std::lock_guard<std::mutex> lock(lane.mutex);
        segment.completed = lane.segment_completed_jobs;
        segment.effects.swap(lane.segment_effects);
        segment.effect_count = segment.effects.size();
        segment.first_task_id = lane.segment_first_task_id;
        segment.last_task_id = lane.segment_last_task_id;
        segment.first_ticket = lane.segment_first_ticket;
        segment.last_ticket = lane.segment_last_ticket;
        segment.all_completed = lane.segment_all_completed;
        segment.has_trap_or_failure = lane.segment_has_trap_or_failure;
        lane.segment_first_task_id = 0u;
        lane.segment_last_task_id = 0u;
        lane.segment_first_ticket = 0u;
        lane.segment_last_ticket = 0u;
        lane.segment_all_completed = false;
        lane.segment_has_trap_or_failure = false;
        lane.segment_result_view = LaneSegmentResultView{};
        lane.segment_result_sequence.store(0u, std::memory_order_relaxed);
        lane.root_reserved.store(false, std::memory_order_release);
      }
    }
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::DeterministicCommitRingPublishes);
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::LaneOwnedSegmentsCompleted);
    ::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::LaneOwnedTasksExecuted) +=
        segment.completed;
    ::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::LaneLocalReadyPops) +=
        segment.completed;
    ::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::LaneLocalEffectPackets) +=
        segment.effect_count;
    ::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::LaneTerminalEffectPackets) +=
        segment.completed;
    ::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::LaneLocalCleanupPackets) +=
        segment.completed;
    ::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::RootPerTaskTerminalScansAvoided) +=
        segment.completed;
    ::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::RootPerTaskCleanupCallsAvoided) +=
        segment.completed;
    ::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::SchedulerContextInstalls) +=
        segment.completed;
    if (record_lane_residual) {
      RecordLaneResidualContextInstalls(state_->evidence.metrics,
                                        segment.completed);
    }
    ::rund::detail::task::Stat(state_->evidence.metrics,
                               ::rund::detail::task::StatSlot::Resumed) +=
        segment.completed;
    summary->commit_logical_events += segment.completed;

    const bool segment_success =
        segment.all_completed && !segment.has_trap_or_failure &&
        segment.completed == segment.jobs.size();
    if (!segment_success) {
      summary->success = false;
      continue;
    }
    if (summary->first_ticket == 0u ||
        segment.first_ticket < summary->first_ticket) {
      summary->first_ticket = segment.first_ticket;
    }
    summary->last_ticket = std::max(summary->last_ticket, segment.last_ticket);
  }
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::LaneCompletionGroupObservedLanes) +=
      completion_group.observed_lanes;
  if (!summary->success && summary->submitted_lanes != 0u) {
    ::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::DeterministicCommitBatches) +=
        summary->submitted_lanes;
    ::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::DeterministicCommitLogicalEvents) +=
        summary->commit_logical_events;
  }
}

} // namespace rund::node
