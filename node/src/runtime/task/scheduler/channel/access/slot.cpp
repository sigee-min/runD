#include "local.hpp"

namespace rund::detail::task {

std::size_t ChannelAccess::TaskSlot(const std::uint64_t task_id) noexcept {
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr || task_id == 0u)
    return std::numeric_limits<std::size_t>::max();
  scheduler->EnsureCurrentCommit();
  return scheduler->state_->IndexFor(task_id);
}

} // namespace rund::detail::task
