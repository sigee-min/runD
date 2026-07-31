#include <rund/task/stats/slots.hpp>

#include "local.hpp"

namespace rund::node {

::rund::detail::task::AwaitDecision Scheduler::Yield() noexcept {
  (void)TrapLaneOwnedSegmentPrimitive(
      ::rund::detail::task::OperationKind::Yield);
  EnsureCurrentCommit();
  TaskRecord *const record = state_->Find(CurrentTaskId());
  if (record == nullptr || record->state != TaskState::Running) {
    return FailYield(ReasonCode::TaskContextMissing);
  }
  if (!record->coroutine_task) {
    SetLeafFailure(*record, ReasonCode::TaskLeafPrimitiveForbidden);
    return FailYield(ReasonCode::TaskLeafPrimitiveForbidden);
  }
  ++::rund::detail::task::Stat(state_->evidence.metrics,
                               ::rund::detail::task::StatSlot::Yields);
  ++::rund::detail::task::Stat(state_->evidence.metrics,
                               ::rund::detail::task::StatSlot::Parked);
  ++::rund::detail::task::Stat(state_->evidence.metrics,
                               ::rund::detail::task::StatSlot::CoroutineParks);
  RecordYieldBatch(record->id);
  record->state = TaskState::Ready;
  record->dynamic_scope_id = CurrentScopeId();
  record->coroutine_parked = true;
  record->lane_segment_side_exit = true;
  state_->EnqueueProgress(*record);
  return ::rund::detail::task::AwaitDecision{.status = task::Status::success(),
                                             .suspend = true};
}

} // namespace rund::node
