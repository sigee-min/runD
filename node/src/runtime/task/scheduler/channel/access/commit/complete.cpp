#include <rund/task/stats/slots.hpp>

#include "../local.hpp"

#include "../../../state/model/context.hpp"

namespace rund::detail::task {

ChannelDecision ChannelAccess::FinishCurrentOperation() noexcept {
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr) {
    return node::channel_access_detail::MissingNodeRuntimeResult();
  }
  scheduler->EnsureCurrentCommit();
  scheduler->CompletePrimitiveCommit();
  return ChannelDecision{.status = ::rund::task::Status::success()};
}

ChannelDecision ChannelAccess::FinishCommittedOperation() noexcept {
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr) {
    return node::channel_access_detail::MissingNodeRuntimeResult();
  }
  node::SchedulerThreadContext *const context = node::active_scheduler_context;
  if (context != nullptr && context->scheduler == scheduler) {
    context->deferred_hot_path_ensure_skips += 2u;
  } else {
    ::rund::detail::task::Stat(
        scheduler->state_->evidence.metrics,
        ::rund::detail::task::StatSlot::SchedulerHotPathEnsureSkips) += 2u;
  }
  scheduler->CompletePrimitiveCommit();
  return ChannelDecision{.status = ::rund::task::Status::success()};
}

ChannelDecision ChannelAccess::FinishCommit(const bool counted) noexcept {
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr) {
    return node::channel_access_detail::MissingNodeRuntimeResult();
  }
  if (counted) {
    node::SchedulerThreadContext *const context =
        node::active_scheduler_context;
    if (context != nullptr && context->scheduler == scheduler) {
      context->deferred_hot_path_ensure_skips += 2u;
    } else {
      ::rund::detail::task::Stat(
          scheduler->state_->evidence.metrics,
          ::rund::detail::task::StatSlot::SchedulerHotPathEnsureSkips) += 2u;
    }
  }
  scheduler->CompletePrimitiveCommit();
  return ChannelDecision{.status = ::rund::task::Status::success()};
}

} // namespace rund::detail::task
