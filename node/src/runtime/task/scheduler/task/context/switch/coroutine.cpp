#include <rund/counter.hpp>
#include <rund/task/stats/slots.hpp>

#include "../local.hpp"

namespace rund::node {

void Scheduler::RunCoroutineQuantum(TaskRecord &record,
                                    const std::uint64_t commit_ticket,
                                    const bool split_primitive_packets,
                                    const bool lane_owned_segment,
                                    const bool root_exclusive_commit,
                                    CompletionLease *const terminal) noexcept {
  SchedulerThreadContext context{
      .scheduler = this,
      .task_id = record.id,
      .scope_id = record.dynamic_scope_id,
      .commit_ticket = commit_ticket,
      .commit_acquired = false,
      .root_submit_recorded = lane_owned_segment,
      .pending_root_submit = false,
      .root_exclusive_commit = root_exclusive_commit,
      .split_primitive_packets = split_primitive_packets,
      .lane_owned_segment_active = lane_owned_segment,
      .lane_owned_segment_trapped = false,
      .deferred_hot_path_ensure_skips = 0u,
      .record = &record,
      .previous = active_scheduler_context,
  };
  active_scheduler_context = &context;
  EnsureCurrentCommit();
  {
    std::lock_guard evidence_lock{state_->evidence.mutex};
    const bool external_wake =
        record.state == TaskState::ExternalBlocked && record.wake_ticket != 0u;
    if (external_wake) {
      record.state = TaskState::Running;
    }
    if (!lane_owned_segment) {
      ++::rund::detail::task::Stat(
          state_->evidence.metrics,
          ::rund::detail::task::StatSlot::SchedulerContextInstalls);
    }
    if (external_wake) {
      record.wake_ticket = 0u;
      record.coroutine_parked = false;
      ++::rund::detail::task::Stat(
          state_->evidence.metrics,
          ::rund::detail::task::StatSlot::ExternalWakes);
      ++::rund::detail::task::Stat(
          state_->evidence.metrics,
          ::rund::detail::task::StatSlot::CoroutineWakes);
      Record(::rund::detail::task::OperationKind::ExternalWake, ReasonCode::Ok,
             record.id);
    }
  }
  if (ResumeCoroutine(record)) {
    if (!lane_owned_segment) {
      FinishQuantum(record);
    }
    FlushDeferredHotPathEnsureSkips(context);
    ReleaseQuantumCommit();
    active_scheduler_context = context.previous;
    return;
  }
  CompletionLease completion{};
  if (!lane_owned_segment) {
    EnsureCurrentCommit();
    std::lock_guard evidence_lock{state_->evidence.mutex};
    completion = state_->resources.completion_pool.lease(record.completion);
    if (completion) {
      const task::Status published = CompletionPool::publish(completion);
      if (!published) {
        record.state = TaskState::Failed;
        record.failure_code = published.code();
      }
    }
    if (record.state == TaskState::Completed) {
      ::rund::detail::counter::Accumulate(
          ::rund::detail::task::Stat(
              state_->evidence.metrics,
              ::rund::detail::task::StatSlot::Completed),
          1u);
      ++::rund::detail::task::Stat(
          state_->evidence.metrics,
          ::rund::detail::task::StatSlot::CoroutineCompletions);
      RecordTerminalBatch(::rund::detail::task::OperationKind::Complete,
                          ReasonCode::Ok, record.id);
      WakeJoinWaiters(record.id, ReasonCode::Ok);
      record.completion = {};
      DestroyTask(record);
    } else if (record.state == TaskState::Failed) {
      ++::rund::detail::task::Stat(state_->evidence.metrics,
                                   ::rund::detail::task::StatSlot::Failed);
      ++::rund::detail::task::Stat(
          state_->evidence.metrics,
          ::rund::detail::task::StatSlot::CoroutineFailures);
      const ReasonCode failure_code = record.failure_code;
      RecordTerminalBatch(::rund::detail::task::OperationKind::Fail,
                          failure_code, record.id);
      WakeJoinWaiters(record.id, failure_code);
      record.completion = {};
      DestroyTask(record);
    }
    record.quantum_active = false;
  }
  FlushDeferredHotPathEnsureSkips(context);
  ReleaseQuantumCommit();
  active_scheduler_context = context.previous;
  if (completion) {
    if (terminal != nullptr) {
      *terminal = completion;
    } else {
      CompletionPool::release(completion);
    }
  }
}

} // namespace rund::node
