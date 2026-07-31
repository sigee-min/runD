#include "local.hpp"

namespace rund::node {

void LaneWorkerAccess::NotifyLaneWaiters(
    TaskLane &lane, const LaneWorkerNotifyFlags flags) noexcept {
  if (flags.signal) {
    lane.completed_job_signal.notify_one();
  }
  if (flags.completion) {
    lane.done_cv.notify_one();
  }
}

void LaneWorkerAccess::PublishSegmentJob(TaskLane &lane,
                                         LaneJobFrame &frame) noexcept {
  LaneWorkerNotifyFlags notify{};
  const bool all_completed =
      frame.segment_all_completed &&
      frame.segment_completed == frame.segment_original_job_count;

  {
    std::lock_guard<std::mutex> lock(lane.mutex);
    lane.segment_jobs.swap(frame.segment_jobs);
    lane.segment_completed_jobs = frame.segment_completed;
    lane.segment_first_task_id = frame.segment_first_task_id;
    lane.segment_last_task_id = frame.segment_last_task_id;
    lane.segment_first_ticket = frame.segment_first_ticket;
    lane.segment_last_ticket = frame.segment_last_ticket;
    lane.segment_all_completed = all_completed;
    lane.segment_has_trap_or_failure = frame.segment_has_trap_or_failure;
    if (frame.segment_result_view_enabled) {
      lane.segment_result_view = LaneSegmentResultView{
          .completed = frame.segment_completed,
          .all_completed = all_completed,
          .has_trap_or_failure = frame.segment_has_trap_or_failure,
      };
      lane.segment_result_sequence.store(frame.job_sequence,
                                         std::memory_order_release);
    }
    lane.segment_job_count = 0u;
    lane.running = false;
    PublishLaneCompletion(lane, frame.job_sequence);
    notify.completion = lane.completion_wait_requested;
    notify.signal = lane.completion_signal_wait_requested;
  }
  NotifyLaneWaiters(lane, notify);
}

void LaneWorkerAccess::ReleaseDirectJob(Scheduler &scheduler, TaskLane &lane,
                                        const bool mailbox) noexcept {
  if (mailbox) {
    lane.mailbox_record.store(nullptr, std::memory_order_relaxed);
    lane.mailbox_task_id.store(0u, std::memory_order_relaxed);
    lane.mailbox_commit_ticket.store(0u, std::memory_order_relaxed);
    lane.mailbox_job_sequence.store(0u, std::memory_order_relaxed);
    lane.mailbox_state.store(0u, std::memory_order_release);
  }
  {
    std::lock_guard<std::mutex> lock(scheduler.state_->batches.direct_mutex);
    if (scheduler.state_->batches.direct_jobs_in_flight > 0u) {
      --scheduler.state_->batches.direct_jobs_in_flight;
    }
    if (scheduler.state_->batches.task_direct_jobs_in_flight > 0u) {
      --scheduler.state_->batches.task_direct_jobs_in_flight;
    }
  }
  scheduler.state_->batches.direct_cv.notify_all();
}

void LaneWorkerAccess::PublishRegularJob(Scheduler &scheduler, TaskLane &lane,
                                         const LaneJobFrame &frame) noexcept {
  LaneWorkerNotifyFlags notify{};
  {
    std::lock_guard<std::mutex> lock(lane.mutex);
    lane.running = false;
    if (!frame.direct_job) {
      PublishLaneCompletion(lane, frame.job_sequence);
      notify.completion = lane.completion_wait_requested;
      notify.signal = lane.completion_signal_wait_requested;
    }
  }

  if (frame.direct_job) {
    ReleaseDirectJob(scheduler, lane, frame.mailbox_job);
    if (frame.completion) {
      CompletionPool::release(frame.completion);
    }
    return;
  }

  NotifyLaneWaiters(lane, notify);
  if (frame.completion) {
    CompletionPool::release(frame.completion);
  }
  if (frame.root_exclusive_hot_standby) {
    const std::uint32_t standby_spins = scheduler.LaneHotStandbySpinBudget();
    for (std::uint32_t spin = 0u; spin < standby_spins; ++spin) {
      if (lane.ready_signal.load(std::memory_order_acquire) !=
          frame.ready_signal_at_job_start) {
        break;
      }
    }
  }
}

void LaneWorkerAccess::Publish(Scheduler &scheduler, TaskLane &lane,
                               LaneJobFrame &frame) noexcept {
  if (frame.segment_job) {
    PublishSegmentJob(lane, frame);
    return;
  }
  if (frame.work != nullptr) {
    {
      std::lock_guard<std::mutex> lock(lane.mutex);
      lane.running = false;
    }
    if (frame.work->released != nullptr) {
      frame.work->released(frame.work->context);
    }
    return;
  }
  PublishRegularJob(scheduler, lane, frame);
}

void PublishLaneJob(Scheduler &scheduler, TaskLane &lane,
                    LaneJobFrame &frame) noexcept {
  LaneWorkerAccess::Publish(scheduler, lane, frame);
}

} // namespace rund::node
