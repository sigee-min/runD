#include "local.hpp"

namespace rund::node {

namespace channel_access_detail {

::rund::detail::task::ChannelDecision MissingNodeRuntimeResult() noexcept {
  return ::rund::detail::task::ChannelDecision{
      .status = task::Status::fail(ReasonCode::NodeRuntimeMissing)};
}

} // namespace channel_access_detail

::rund::detail::task::ActiveState Scheduler::ActiveState() noexcept {
  EnsureCurrentCommit();
  return ::rund::detail::task::ActiveState{
      .active = true,
      .scheduler_id = state_->identity.scheduler_id,
      .task_id = CurrentTaskId(),
      .task_slot = state_->IndexFor(CurrentTaskId()),
      .task_capacity = state_->resources.limits.task_capacity,
  };
}

} // namespace rund::node

namespace rund::detail::task {

ActiveState ChannelAccess::ActiveSchedulerState() noexcept {
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr) {
    return ActiveState{};
  }
  return scheduler->ActiveState();
}

} // namespace rund::detail::task
