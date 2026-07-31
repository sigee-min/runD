#include "../local.hpp"

#include "../../../state/model/context.hpp"
#include "../../../state/model/task.hpp"

namespace rund::detail::task {

ChannelDecision ChannelAccess::CommitSchedulerOperationLight(
    const ::rund::detail::task::OperationKind operation_kind,
    std::uint64_t *const scheduler_id, std::uint64_t *const task_id) noexcept {
  if (scheduler_id != nullptr) {
    *scheduler_id = 0u;
  }
  if (task_id != nullptr) {
    *task_id = 0u;
  }
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr) {
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
      if (scheduler_id != nullptr) {
        *scheduler_id = scheduler->state_->identity.scheduler_id;
      }
      if (task_id != nullptr) {
        *task_id = scheduler->CurrentTaskId();
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
  if (scheduler_id != nullptr) {
    *scheduler_id = scheduler->state_->identity.scheduler_id;
  }
  if (task_id != nullptr) {
    *task_id = scheduler->CurrentTaskId();
  }
  return ChannelDecision{.status = ::rund::task::Status::success()};
}

} // namespace rund::detail::task
