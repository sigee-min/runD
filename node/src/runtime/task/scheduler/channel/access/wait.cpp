#include "local.hpp"

namespace rund::detail::task {

ChannelDecision ChannelAccess::ParkChannelWait(const std::uint64_t channel_id,
                                               const bool send_wait) noexcept {
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr) {
    return node::channel_access_detail::MissingNodeRuntimeResult();
  }
  return scheduler->ParkChannel(channel_id, send_wait);
}

ChannelDecision
ChannelAccess::WakeChannelTask(const std::uint64_t task_id,
                               const std::uint64_t channel_id) noexcept {
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr) {
    return node::channel_access_detail::MissingNodeRuntimeResult();
  }
  return scheduler->WakeChannel(task_id, channel_id);
}

} // namespace rund::detail::task
