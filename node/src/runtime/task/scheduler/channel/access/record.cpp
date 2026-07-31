#include "local.hpp"

namespace rund::detail::task {

ChannelDecision
ChannelAccess::RecordChannelSend(const std::uint64_t channel_id,
                                 const std::uint64_t value_count) noexcept {
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr) {
    return node::channel_access_detail::MissingNodeRuntimeResult();
  }
  return scheduler->RecordChannel(
      ::rund::detail::task::OperationKind::ChannelSend, channel_id,
      value_count);
}

ChannelDecision ChannelAccess::RecordBufferedChannelSendBatch(
    const std::uint64_t channel_id, const std::uint64_t value_count,
    const std::uint64_t logical_sends) noexcept {
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr) {
    return node::channel_access_detail::MissingNodeRuntimeResult();
  }
  return scheduler->RecordBufferedChannelSendBatch(channel_id, value_count,
                                                   logical_sends);
}

ChannelDecision
ChannelAccess::RecordChannelRecv(const std::uint64_t channel_id,
                                 const std::uint64_t value_count) noexcept {
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr) {
    return node::channel_access_detail::MissingNodeRuntimeResult();
  }
  return scheduler->RecordChannel(
      ::rund::detail::task::OperationKind::ChannelRecv, channel_id,
      value_count);
}

ChannelDecision
ChannelAccess::RecordChannelClose(const std::uint64_t channel_id) noexcept {
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr) {
    return node::channel_access_detail::MissingNodeRuntimeResult();
  }
  return scheduler->RecordChannel(
      ::rund::detail::task::OperationKind::ChannelClose, channel_id, 0u);
}

void ChannelAccess::RecordCommittedChannelClose(
    const std::uint64_t channel_id) noexcept {
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr) {
    return;
  }
  scheduler->RecordCommittedChannel(
      ::rund::detail::task::OperationKind::ChannelClose, channel_id, 0u);
}

} // namespace rund::detail::task
