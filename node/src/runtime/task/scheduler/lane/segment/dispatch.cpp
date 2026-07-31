#include <rund/task/stats/slots.hpp>

#include "../../state/model/lane.hpp"
#include "../../state/segment.hpp"
#include "../../state/storage.hpp"
#include "../residual.hpp"
#include "transaction.hpp"

namespace rund::node {

std::size_t Scheduler::PublishLaneOwnedSegments(
    std::vector<LaneOwnedSegmentLane> &segments,
    const std::size_t planned_logical_tasks) noexcept {
  std::size_t planned_lanes = 0u;
  std::size_t logical_tasks = 0u;
  for (const LaneOwnedSegmentLane &segment : segments) {
    if (!segment.jobs.empty()) {
      ++planned_lanes;
      logical_tasks += segment.jobs.size();
      if (segment.effects.capacity() < segment.jobs.size()) {
        return 0u;
      }
      for (const LaneSegmentJob &job : segment.jobs) {
        if (job.canonical_index >= planned_logical_tasks) {
          return 0u;
        }
      }
    }
  }
  if (planned_lanes == 0u || logical_tasks != planned_logical_tasks) {
    return 0u;
  }
  const bool publish_result_views = planned_lanes != 0u;
  const bool published = segment::Publish(
      state_->lanes.lanes,
      [&segments](const std::size_t index) noexcept {
        return !segments[index].jobs.empty();
      },
      [&, this]() noexcept {
        const std::uint64_t first_ticket =
            IssueLaneCommitTickets(planned_logical_tasks);
        if (first_ticket == 0u) {
          return false;
        }
        for (LaneOwnedSegmentLane &planned : segments) {
          if (planned.jobs.empty()) {
            continue;
          }
          for (LaneSegmentJob &job : planned.jobs) {
            job.logical_ticket = first_ticket + job.canonical_index;
          }
          planned.first_ticket = planned.jobs.front().logical_ticket;
          planned.last_ticket = planned.jobs.back().logical_ticket;
        }

        for (std::size_t lane_index = 0u; lane_index < segments.size();
             ++lane_index) {
          LaneOwnedSegmentLane &planned = segments[lane_index];
          if (planned.jobs.empty()) {
            continue;
          }
          TaskLane &lane = *state_->lanes.lanes[lane_index];
          lane.segment_effects.clear();
          lane.segment_first_task_id = 0u;
          lane.segment_last_task_id = 0u;
          lane.segment_first_ticket = 0u;
          lane.segment_last_ticket = 0u;
          lane.segment_all_completed = false;
          lane.segment_has_trap_or_failure = false;
          lane.segment_result_view = LaneSegmentResultView{};
          lane.segment_result_sequence.store(0u, std::memory_order_relaxed);
          lane.segment_jobs.swap(planned.jobs);
          lane.segment_effects.swap(planned.effects);
          lane.segment_job_count = lane.segment_jobs.size();
          lane.segment_completed_jobs = 0u;
          planned.sequence = lane.next_job_sequence++;
          lane.job_sequence = planned.sequence;
          lane.task_id = lane.segment_jobs.front().task_id;
          lane.task_record = nullptr;
          lane.commit_ticket = 0u;
          lane.split_primitive_packets = true;
          lane.direct_job = false;
          lane.root_exclusive_hot_standby = false;
          lane.root_reserved.store(true, std::memory_order_release);
          lane.completion_wait_requested = false;
          lane.completion_signal_wait_requested = false;
          lane.segment_result_view_enabled = publish_result_views;
          lane.has_job = true;
          lane.ready_signal.store(planned.sequence, std::memory_order_release);
          planned.notify_ready = lane.ready_wait_requested;
          planned.submitted = true;
        }
        return true;
      });
  if (!published) {
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::CrossLaneConflicts);
    return 0u;
  }

  for (std::size_t lane_index = 0u; lane_index < segments.size();
       ++lane_index) {
    if (segments[lane_index].submitted &&
        lane_index < state_->lanes.lane_participated.size() &&
        state_->lanes.lane_participated[lane_index] == 0u) {
      state_->lanes.lane_participated[lane_index] = 1u;
      ++::rund::detail::task::Stat(
          state_->evidence.metrics,
          ::rund::detail::task::StatSlot::ParticipatingTaskWorkers);
    }
  }
  return planned_lanes;
}

void Scheduler::RecordLaneOwnedSegmentAdmission(
    const std::size_t submitted_lanes,
    const std::size_t logical_tasks) noexcept {
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::RootPolicyInstalls);
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::LaneOrderHashFastPaths);
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::LaneDispatchBatchPackets);
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::RootDispatchProofBatches);
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::LaneDispatchBatchLogicalTasks) +=
      logical_tasks;
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::LaneOwnedSegmentsStarted) +=
      submitted_lanes;
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::LaneLocalReadyPushes) += logical_tasks;
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::LaneSegmentRecordLookupAvoids) +=
      logical_tasks;
  if (logical_tasks > submitted_lanes) {
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::DeterministicWorkShareTicketRanges);
    ::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::DeterministicWorkShareLogicalTasks) +=
        logical_tasks - submitted_lanes;
    ::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::RootPerTaskDispatchesAvoided) +=
        logical_tasks - submitted_lanes;
    ::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::RootPerTaskCompletionDrainsAvoided) +=
        logical_tasks - submitted_lanes;
  }
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::RootPerTaskTicketReservationsAvoided) +=
      logical_tasks - 1u;
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::RootTicketReservations);
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::LaneSchedulerShardsActive) +=
      submitted_lanes;
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::LaneTicketRangesReserved);
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::LaneTicketRangeLogicalEvents) +=
      logical_tasks;
}

void Scheduler::NotifyLaneOwnedSegments(
    const std::vector<LaneOwnedSegmentLane> &segments,
    const std::size_t submitted_lanes) noexcept {
  const bool record_lane_residual =
      ShouldRecordLaneResidualSegmentMetrics(submitted_lanes);
  for (std::size_t lane_index = 0u; lane_index < segments.size();
       ++lane_index) {
    if (segments[lane_index].submitted) {
      if (record_lane_residual) {
        RecordLaneResidualWakeNotifications(state_->evidence.metrics, 1u);
      }
      if (segments[lane_index].notify_ready) {
        ++::rund::detail::task::Stat(
            state_->evidence.metrics,
            ::rund::detail::task::StatSlot::LaneReadySingleWakes);
        state_->lanes.lanes[lane_index]->ready.notify_one();
      }
    }
  }
}

} // namespace rund::node
