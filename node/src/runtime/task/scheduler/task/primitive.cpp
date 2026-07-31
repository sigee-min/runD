#include "../state/model/context.hpp"
#include "../state/model/segment.hpp"
#include "../state/model/task.hpp"
#include "../state/storage.hpp"

namespace rund::node {

void Scheduler::SetLeafFailure(TaskRecord &record,
                               const ReasonCode code) noexcept {
  SchedulerThreadContext *const context = active_scheduler_context;
  if (context != nullptr && context->scheduler == this &&
      context->lane_effect != nullptr) {
    LaneSegmentEffect &effect = *context->lane_effect;
    effect.terminal = true;
    effect.terminal_kind = ::rund::detail::task::OperationKind::Fail;
    effect.code = code;
    effect.trapped = false;
    return;
  }
  std::lock_guard evidence_lock{state_->evidence.mutex};
  record.state = TaskState::Failed;
  record.failure_code = code;
}

ReasonCode Scheduler::RejectPrimitive() noexcept {
  EnsureCurrentCommit();
  TaskRecord *const active = state_->Find(CurrentTaskId());
  const bool coroutine = active != nullptr && active->coroutine_task;
  const ReasonCode code = coroutine
                              ? ReasonCode::TaskCoroutinePrimitiveAwaitNotLive
                              : ReasonCode::TaskLeafPrimitiveForbidden;
  if (active != nullptr && !coroutine) {
    SetLeafFailure(*active, code);
  }
  CompletePrimitiveCommit();
  return code;
}

} // namespace rund::node
