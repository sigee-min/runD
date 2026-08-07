#include <rund/counter.hpp>
#include <rund/task/stats/slots.hpp>

#include "../local.hpp"
#include "../../../state/model/segment.hpp"

namespace rund::node {

void Scheduler::RunLeafQuantum(TaskRecord &record,
                               const std::uint64_t commit_ticket,
                               const bool split_primitive_packets,
                               const bool lane_owned_segment,
                               const bool root_exclusive_commit,
                               LaneSegmentEffect *const lane_effect) noexcept {
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
      .lane_effect = lane_effect,
      .deferred_hot_path_ensure_skips = 0u,
      .record = &record,
      .previous = active_scheduler_context,
  };
  active_scheduler_context = &context;
  ReasonCode result = ReasonCode::Ok;
  try {
    if (record.callable) {
      (*record.callable)();
    } else {
      result = ReasonCode::TaskInvalid;
    }
  } catch (...) {
    result = ReasonCode::TaskFailed;
  }

  // Calculation may run on any physical lane. Observable state, evidence and
  // terminal effects are published only at the task's canonical commit turn.
  if (!lane_owned_segment) {
    EnsureCurrentCommit();
  } else if (lane_effect != nullptr && !lane_effect->terminal) {
    lane_effect->terminal = true;
    lane_effect->terminal_kind =
        result == ReasonCode::Ok
            ? ::rund::detail::task::OperationKind::Complete
            : ::rund::detail::task::OperationKind::Fail;
    lane_effect->code = result;
  }
  if (!lane_owned_segment) {
    std::lock_guard evidence_lock{state_->evidence.mutex};
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::SchedulerContextInstalls);
    if (record.state == TaskState::Running) {
      record.state = result == ReasonCode::Ok ? TaskState::Completed
                                              : TaskState::Failed;
      record.failure_code = result;
    }
    if (record.state == TaskState::Completed) {
      ::rund::detail::counter::Accumulate(
          ::rund::detail::task::Stat(
              state_->evidence.metrics,
              ::rund::detail::task::StatSlot::Completed),
          1u);
      RecordTerminalBatch(::rund::detail::task::OperationKind::Complete,
                          ReasonCode::Ok, record.id);
      WakeJoinWaiters(record.id, ReasonCode::Ok);
      DestroyTask(record);
    } else {
      ++::rund::detail::task::Stat(state_->evidence.metrics,
                                   ::rund::detail::task::StatSlot::Failed);
      const ReasonCode code = record.failure_code;
      RecordTerminalBatch(::rund::detail::task::OperationKind::Fail, code,
                          record.id);
      WakeJoinWaiters(record.id, code);
      DestroyTask(record);
    }
    record.quantum_active = false;
  }
  FlushDeferredHotPathEnsureSkips(context);
  if (lane_owned_segment && !context.commit_acquired) {
    CompleteLaneCommit(context.commit_ticket);
  } else {
    ReleaseQuantumCommit();
  }
  active_scheduler_context = context.previous;
}

} // namespace rund::node
