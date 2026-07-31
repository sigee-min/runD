#include <rund/task/stats/slots.hpp>

#include "../../state/model/context.hpp"
#include "../../state/storage.hpp"

namespace rund::node {

void Scheduler::FlushPendingRootSubmit() noexcept {
  SchedulerThreadContext *const context = active_scheduler_context;
  if (context == nullptr || context->scheduler != this ||
      !context->pending_root_submit) {
    return;
  }
  FlushYieldBatch(ReasonCode::Ok);
  RecordPhysical(::rund::detail::task::OperationKind::RootSubmit,
                 ReasonCode::Ok, context->task_id);
  context->pending_root_submit = false;
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::SchedulerResumeRootSubmitFlushed);
}

bool Scheduler::ConsumePendingRootSubmit(const std::uint64_t task_id) noexcept {
  SchedulerThreadContext *const context = active_scheduler_context;
  if (context == nullptr || context->scheduler != this ||
      !context->pending_root_submit || context->task_id != task_id) {
    return false;
  }
  context->pending_root_submit = false;
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::SchedulerResumeRootSubmitBatched);
  return true;
}

} // namespace rund::node
