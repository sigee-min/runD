#include "local.hpp"

#include "../../state/model/context.hpp"
#include "../../state/model/task.hpp"

namespace rund::detail::task {

ChannelDecision ChannelAccess::CommitSchedulerOperation(
    const ::rund::detail::task::OperationKind operation_kind) noexcept {
  return CommitSchedulerOperation(operation_kind, nullptr);
}

ChannelDecision ChannelAccess::CommitSchedulerOperation(
    const ::rund::detail::task::OperationKind operation_kind,
    ActiveState *const active_state) noexcept {
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr) {
    if (active_state != nullptr) {
      *active_state = ActiveState{};
    }
    return node::channel_access_detail::MissingNodeRuntimeResult();
  }
  scheduler->EnsureCurrentCommit();
  (void)scheduler->TrapLaneOwnedSegmentPrimitive(operation_kind);
  if (scheduler->CurrentTaskId() != 0u) {
    node::SchedulerThreadContext *const context =
        node::active_scheduler_context;
    node::TaskRecord *const record =
        context != nullptr && context->scheduler == scheduler
            ? static_cast<node::TaskRecord *>(context->record)
            : nullptr;
    if (record != nullptr && record->coroutine_task) {
      if (active_state != nullptr) {
        *active_state = scheduler->ActiveState();
      }
      return ChannelDecision{.status = ::rund::task::Status::success()};
    }
    if (record != nullptr) {
      scheduler->SetLeafFailure(*record,
                                ReasonCode::TaskLeafPrimitiveForbidden);
    }
    return ChannelDecision{.status = ::rund::task::Status::fail(
                               ReasonCode::TaskLeafPrimitiveForbidden)};
  }
  if (active_state != nullptr) {
    *active_state = scheduler->ActiveState();
  }
  return ChannelDecision{.status = ::rund::task::Status::success()};
}

} // namespace rund::detail::task
