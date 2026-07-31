#include <rund/task/stats/slots.hpp>

#include "../state/model/context.hpp"
#include "../state/model/lane.hpp"
#include "../state/model/task.hpp"
#include "../state/storage.hpp"

namespace rund::node {

bool Scheduler::CanSubmitToLane(const std::uint64_t id) noexcept {
  if (state_->lanes.lanes.empty()) {
    return false;
  }
  const std::size_t lane_index =
      state_->LaneIndexForTask(id, state_->lanes.lanes.size());
  if (lane_index >= state_->lanes.lanes.size()) {
    return false;
  }
  TaskLane &lane = *state_->lanes.lanes[lane_index];
  if (active_task_lane == &lane) {
    return true;
  }
  std::lock_guard<std::mutex> lock(lane.mutex);
  const bool idle_completion =
      lane.completed_job_sequence.load(std::memory_order_acquire) == 0u;
  const bool root_reserved = lane.root_reserved.load(std::memory_order_acquire);
  const bool nested_accept = lane.accepts_nested_job && !root_reserved &&
                             !lane.has_job && !lane.nested_job_active &&
                             idle_completion;
  return !lane.stop && !lane.has_job && !root_reserved && idle_completion &&
         (!lane.running || nested_accept);
}

bool Scheduler::DispatchQueuedReady(const std::uint64_t id) noexcept {
  if (state_->lanes.lanes.empty()) {
    return false;
  }
  TaskRecord *record = state_->Find(id);
  if (record == nullptr || record->state != TaskState::Ready ||
      record->coroutine_task || record->lane_segment_side_exit) {
    return false;
  }
  const std::size_t lane_index =
      state_->LaneIndexForTask(record, state_->lanes.lanes.size());
  if (lane_index >= state_->lanes.lanes.size()) {
    return false;
  }
  TaskLane &lane = *state_->lanes.lanes[lane_index];
  bool notify = false;
  {
    std::lock_guard<std::mutex> lock(lane.mutex);
    if (lane.stop) {
      return false;
    }
    record = PrepareLaneQuantum(*record);
    if (record == nullptr) {
      return false;
    }
    const std::uint64_t ticket = IssueCommitTicket();
    const std::uint64_t sequence = lane.next_job_sequence++;
    QueueDirect(lane, *record, ticket, sequence);
    lane.ready_signal.store(sequence, std::memory_order_release);
    notify = lane.ready_wait_requested;
  }
  if (lane_index < state_->lanes.lane_participated.size() &&
      state_->lanes.lane_participated[lane_index] == 0u) {
    state_->lanes.lane_participated[lane_index] = 1u;
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::ParticipatingTaskWorkers);
  }
  if (notify) {
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::LaneReadySingleWakes);
    lane.ready.notify_one();
  }
  return true;
}

bool Scheduler::RunOnLane(const std::uint64_t id,
                          const std::uint64_t commit_ticket,
                          const bool split_primitive_packets,
                          const bool root_exclusive_commit,
                          const bool root_exclusive_hot_standby) noexcept {
  if (state_->lanes.lanes.empty()) {
    TaskRecord *const record = state_->Find(id);
    if (record != nullptr) {
      record->state = TaskState::Failed;
      record->failure_code = ReasonCode::TaskWorkersInvalid;
    }
    return false;
  }
  const std::size_t lane_index =
      state_->LaneIndexForTask(id, state_->lanes.lanes.size());
  if (lane_index >= state_->lanes.lanes.size()) {
    return false;
  }
  TaskLane &lane = *state_->lanes.lanes[lane_index];
  if (active_task_lane == &lane) {
    TaskRecord *const existing_record = state_->Find(id);
    if (existing_record == nullptr) {
      return true;
    }
    TaskRecord *const record = PrepareLaneQuantum(id);
    if (record == nullptr) {
      return true;
    }
    const bool issue_ticket = commit_ticket == 0u &&
                              active_scheduler_context == nullptr &&
                              !root_exclusive_commit;
    const std::uint64_t ticket =
        issue_ticket ? IssueCommitTicket() : commit_ticket;
    RunTaskQuantum(*record, ticket, split_primitive_packets, false,
                   root_exclusive_commit);
    return true;
  }
  std::uint64_t job_sequence = 0u;
  TaskRecord *record = nullptr;
  std::uint64_t ticket = 0u;
  bool notify_ready_waiter = false;
  {
    std::lock_guard<std::mutex> lock(lane.mutex);
    const bool idle_completion =
        lane.completed_job_sequence.load(std::memory_order_acquire) == 0u;
    const bool root_reserved =
        lane.root_reserved.load(std::memory_order_acquire);
    const bool nested_accept = lane.accepts_nested_job && !root_reserved &&
                               !lane.has_job && !lane.nested_job_active &&
                               idle_completion;
    if (lane.stop || lane.has_job || root_reserved || !idle_completion ||
        (lane.running && !nested_accept)) {
      return false;
    }
    record = state_->Find(id);
    if (record == nullptr) {
      return true;
    }
    record = PrepareLaneQuantum(id);
    if (record == nullptr) {
      return true;
    }
    const bool issue_ticket = commit_ticket == 0u &&
                              active_scheduler_context == nullptr &&
                              !root_exclusive_commit;
    ticket = issue_ticket ? IssueCommitTicket() : commit_ticket;
    job_sequence = lane.next_job_sequence++;
    lane.task_id = id;
    lane.task_record = record;
    lane.commit_ticket = ticket;
    lane.job_sequence = job_sequence;
    lane.split_primitive_packets = split_primitive_packets;
    lane.root_exclusive_commit = root_exclusive_commit;
    lane.direct_job = false;
    lane.root_exclusive_hot_standby = root_exclusive_hot_standby;
    lane.root_reserved.store(false, std::memory_order_release);
    lane.completion_wait_requested = false;
    lane.completion_signal_wait_requested = false;
    lane.has_job = true;
    lane.ready_signal.store(job_sequence, std::memory_order_release);
    notify_ready_waiter = lane.ready_wait_requested;
  }
  if (root_exclusive_hot_standby) {
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::LaneReadyHotStandbySpins);
    if (!notify_ready_waiter) {
      ++::rund::detail::task::Stat(
          state_->evidence.metrics,
          ::rund::detail::task::StatSlot::LaneReadyHotStandbyHits);
    }
  }
  if (lane_index < state_->lanes.lane_participated.size() &&
      state_->lanes.lane_participated[lane_index] == 0u) {
    state_->lanes.lane_participated[lane_index] = 1u;
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::ParticipatingTaskWorkers);
  }
  if (notify_ready_waiter) {
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::LaneReadySingleWakes);
    lane.ready.notify_one();
  }
  if (active_task_lane != nullptr) {
    TaskLane &own_lane = *active_task_lane;
    const auto clear_nested_accept = [&own_lane] {
      std::lock_guard<std::mutex> own_lock(own_lane.mutex);
      own_lane.accepts_nested_job = false;
    };
    {
      std::lock_guard<std::mutex> own_lock(own_lane.mutex);
      own_lane.accepts_nested_job = true;
    }
    for (;;) {
      {
        std::unique_lock<std::mutex> target_lock(lane.mutex);
        if (TakeLaneCompletion(lane, job_sequence)) {
          lane.root_reserved.store(false, std::memory_order_release);
          clear_nested_accept();
          return true;
        }
        if (lane.completed_job_sequence.load(std::memory_order_acquire) !=
            job_sequence) {
          lane.completion_wait_requested = true;
        }
        (void)lane.done_cv.wait_for(
            target_lock, NestedJoinPollInterval(), [&lane, job_sequence] {
              return lane.completed_job_sequence.load(
                         std::memory_order_acquire) == job_sequence;
            });
        if (TakeLaneCompletion(lane, job_sequence)) {
          lane.root_reserved.store(false, std::memory_order_release);
          clear_nested_accept();
          return true;
        }
      }

      std::uint64_t help_id = 0u;
      TaskRecord *help_record = nullptr;
      std::uint64_t help_ticket = 0u;
      std::uint64_t help_sequence = 0u;
      bool help_split_primitive_packets = false;
      bool help_root_exclusive_commit = false;
      {
        std::lock_guard<std::mutex> own_lock(own_lane.mutex);
        if (own_lane.has_job) {
          help_id = own_lane.task_id;
          help_record = own_lane.task_record;
          help_ticket = own_lane.commit_ticket;
          help_sequence = own_lane.job_sequence;
          help_split_primitive_packets = own_lane.split_primitive_packets;
          help_root_exclusive_commit = own_lane.root_exclusive_commit;
          own_lane.direct_job = false;
          own_lane.has_job = false;
          own_lane.nested_job_active = true;
          own_lane.root_exclusive_commit = false;
        }
      }
      if (help_id == 0u) {
        continue;
      }
      if (help_record != nullptr) {
        RunTaskQuantum(*help_record, help_ticket, help_split_primitive_packets,
                       false, help_root_exclusive_commit);
      }
      bool notify_completion_waiter = false;
      {
        std::lock_guard<std::mutex> own_lock(own_lane.mutex);
        PublishLaneCompletion(own_lane, help_sequence);
        own_lane.nested_job_active = false;
        notify_completion_waiter = own_lane.completion_wait_requested;
      }
      if (notify_completion_waiter) {
        own_lane.done_cv.notify_one();
      }
    }
  }
  if (TryConsumeCompletedJobSpin(lane, job_sequence, true)) {
    return true;
  }
  (void)WaitForLaneCompletionSignal(lane, job_sequence);
  std::lock_guard<std::mutex> lock(lane.mutex);
  lane.root_reserved.store(false, std::memory_order_release);
  return true;
}

} // namespace rund::node
