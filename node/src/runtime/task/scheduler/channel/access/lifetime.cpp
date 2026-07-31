#include "local.hpp"

namespace rund::detail::task {

ChannelDecision
ChannelAccess::MakeChannelRecord(const std::size_t capacity,
                                 std::uint64_t *const out_channel_id) noexcept {
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr) {
    return node::channel_access_detail::MissingNodeRuntimeResult();
  }
  return scheduler->MakeChannel(capacity, out_channel_id);
}

ChannelDecision
ChannelAccess::ReleaseChannelRecord(const std::uint64_t channel_id,
                                    const std::size_t capacity) noexcept {
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr) {
    return node::channel_access_detail::MissingNodeRuntimeResult();
  }
  return scheduler->ReleaseChannel(channel_id, capacity);
}

void ChannelAccess::ReleaseCommittedChannelRecord(
    const std::uint64_t channel_id, const std::size_t capacity) noexcept {
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr) {
    return;
  }
  scheduler->ReleaseCommittedChannel(channel_id, capacity);
}

} // namespace rund::detail::task
