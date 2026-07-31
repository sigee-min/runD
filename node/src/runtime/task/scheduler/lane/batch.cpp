#include <rund/task/stats/slots.hpp>

#include "../state/model/batch.hpp"
#include "../state/model/context.hpp"
#include "../state/model/lane.hpp"
#include "../state/model/task.hpp"
#include "../state/storage.hpp"
#include "segment/transaction.hpp"

#include <algorithm>
#include <cstdlib>
#include <thread>

namespace rund::node {

bool Scheduler::RunReadyBatch(
    const std::vector<std::uint64_t> &task_ids) noexcept {
  if (task_ids.empty()) {
    return false;
  }
  if (task_ids.size() == 1u) {
    return RunOnLane(task_ids.front());
  }
  const auto requeue_from = [this, &task_ids](const std::size_t first) noexcept {
    for (std::size_t reverse = task_ids.size(); reverse > first; --reverse) {
      (void)RequeueReadyTask(task_ids[reverse - 1u], 0u);
    }
  };
  if (active_scheduler_context != nullptr) {
    requeue_from(1u);
    return RunOnLane(task_ids.front());
  }
  std::vector<LaneBatchSubmission> &submitted =
      state_->lanes.submitted_batch_scratch;
  std::vector<std::uint8_t> &active =
      state_->lanes.lane_batch_used_scratch;
  if (submitted.capacity() < task_ids.size() ||
      active.size() != state_->lanes.lanes.size()) {
    std::abort();
  }
  submitted.clear();
  std::fill(active.begin(), active.end(), std::uint8_t{0u});
  std::size_t consumed = 0u;
  for (; consumed < task_ids.size(); ++consumed) {
    const std::uint64_t id = task_ids[consumed];
    TaskRecord *const record = state_->Find(id);
    if (record == nullptr || record->state == TaskState::Completed ||
        record->state == TaskState::Failed) {
      continue;
    }
    if (record->state != TaskState::Ready) {
      std::abort();
    }
    const std::size_t lane_index =
        state_->LaneIndexForTask(record, state_->lanes.lanes.size());
    if (lane_index >= active.size()) {
      std::abort();
    }
    if (active[lane_index] != 0u) {
      break;
    }
    active[lane_index] = 1u;
    TaskLane &lane = *state_->lanes.lanes[lane_index];
    submitted.push_back(
        LaneBatchSubmission{.lane = &lane,
                            .record = record,
                            .task_id = id,
                            .ticket = 0u,
                            .sequence = 0u,
                            .split_primitive_packets = false});
  }
  if (submitted.empty()) {
    requeue_from(consumed);
    return true;
  }
  const bool split = submitted.size() > 1u;
  for (LaneBatchSubmission &job : submitted) {
    job.split_primitive_packets = split;
  }
  const bool published = segment::Publish(
      state_->lanes.lanes,
      [&active](const std::size_t index) noexcept {
        return active[index] != 0u;
      },
      [&, this]() noexcept {
        for (LaneBatchSubmission &job : submitted) {
          job.record = PrepareLaneQuantum(job.task_id);
          if (job.record == nullptr) {
            std::abort();
          }
        }
        const std::uint64_t first_ticket =
            IssueCommitTickets(submitted.size());
        if (first_ticket == 0u) {
          std::abort();
        }
        for (std::size_t index = 0u; index < submitted.size(); ++index) {
          LaneBatchSubmission &job = submitted[index];
          TaskLane &lane = *job.lane;
          job.ticket = first_ticket + index;
          job.sequence = lane.next_job_sequence++;
          lane.task_id = job.task_id;
          lane.task_record = job.record;
          lane.commit_ticket = job.ticket;
          lane.job_sequence = job.sequence;
          lane.split_primitive_packets = job.split_primitive_packets;
          lane.direct_job = false;
          lane.root_exclusive_hot_standby = false;
          lane.root_reserved.store(true, std::memory_order_release);
          lane.completion_wait_requested = false;
          lane.completion_signal_wait_requested = false;
          lane.has_job = true;
          lane.ready_signal.store(job.sequence, std::memory_order_release);
          job.notify_ready = lane.ready_wait_requested;
        }
        return true;
      });
  if (!published) {
    requeue_from(0u);
    std::this_thread::yield();
    return true;
  }
  requeue_from(consumed);

  if (split) {
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::LaneDispatchBatchPackets);
    ::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::LaneDispatchBatchLogicalTasks) +=
        submitted.size();
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::RootDispatchProofBatches);
  }
  for (LaneBatchSubmission &job : submitted) {
    const std::size_t lane_index = state_->LaneIndexForTask(
        job.record, state_->lanes.lanes.size());
    if (lane_index < state_->lanes.lane_participated.size() &&
        state_->lanes.lane_participated[lane_index] == 0u) {
      state_->lanes.lane_participated[lane_index] = 1u;
      ++::rund::detail::task::Stat(
          state_->evidence.metrics,
          ::rund::detail::task::StatSlot::ParticipatingTaskWorkers);
    }
    if (job.notify_ready) {
      ++::rund::detail::task::Stat(
          state_->evidence.metrics,
          ::rund::detail::task::StatSlot::LaneReadySingleWakes);
      job.lane->ready.notify_one();
    }
  }
  for (LaneBatchSubmission &job : submitted) {
    if (TryConsumeCompletedJobSpin(*job.lane, job.sequence)) {
      ++::rund::detail::task::Stat(
          state_->evidence.metrics,
          ::rund::detail::task::StatSlot::LaneLocalCompletionPackets);
      continue;
    }
    (void)WaitForLaneCompletionSignal(*job.lane, job.sequence);
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::LaneLocalCompletionPackets);
  }
  for (LaneBatchSubmission &job : submitted) {
    std::lock_guard<std::mutex> lock(job.lane->mutex);
    job.lane->root_reserved.store(false, std::memory_order_release);
  }
  return true;
}

} // namespace rund::node
