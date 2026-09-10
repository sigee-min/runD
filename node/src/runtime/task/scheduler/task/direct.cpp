#include "../state/model/context.hpp"
#include "../state/model/lane.hpp"
#include "../state/model/task.hpp"
#include "../state/storage.hpp"

namespace rund::node {

bool Scheduler::TrapLaneOwnedSegmentPrimitive(
    const ::rund::detail::task::OperationKind kind,
    const ReasonCode code) noexcept {
  SchedulerThreadContext *const context = active_scheduler_context;
  if (context == nullptr || context->scheduler != this ||
      !context->lane_owned_segment_active ||
      context->lane_owned_segment_trapped) {
    return false;
  }
  TaskRecord *const record = static_cast<TaskRecord *>(context->record);
  if (record == nullptr || record->state != TaskState::Running) {
    return false;
  }
  context->lane_owned_segment_active = false;
  context->lane_owned_segment_trapped = true;
  if (context->lane_effect != nullptr) {
    context->lane_effect->trapped = true;
    context->lane_effect->trap_kind = kind;
    context->lane_effect->trap_code = code;
  }
  return true;
}

} // namespace rund::node
