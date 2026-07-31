#include <rund/task/stats/slots.hpp>

#include "../state/storage.hpp"

#include <limits>

namespace rund::node {

::rund::detail::task::ChannelDecision
Scheduler::MakeChannel(const std::size_t capacity,
                       std::uint64_t *const out_channel_id) noexcept {
  (void)TrapLaneOwnedSegmentPrimitive(
      ::rund::detail::task::OperationKind::ChannelMake);
  if (out_channel_id == nullptr) {
    return ::rund::detail::task::ChannelDecision{
        .status = task::Status::fail(ReasonCode::ChannelInvalid)};
  }
  EnsureCurrentCommit();
  if (state_->resources.live_channels >=
      state_->resources.limits.channel_capacity) {
    return ::rund::detail::task::ChannelDecision{
        .status = task::Status::fail(ReasonCode::ChannelCapacityExceeded)};
  }
  if (capacity > std::numeric_limits<std::uint32_t>::max() ||
      capacity > state_->resources.limits.channel_buffer_capacity ||
      state_->resources.live_channel_buffer_slots >
          state_->resources.limits.channel_buffer_capacity -
              static_cast<std::uint32_t>(capacity)) {
    return ::rund::detail::task::ChannelDecision{
        .status =
            task::Status::fail(ReasonCode::ChannelBufferCapacityExceeded)};
  }
  *out_channel_id = state_->identity.next_channel_id++;
  ++state_->resources.live_channels;
  state_->resources.live_channel_buffer_slots +=
      static_cast<std::uint32_t>(capacity);
  Record(::rund::detail::task::OperationKind::ChannelMake, ReasonCode::Ok,
         CurrentTaskId(), 0u, 0u, *out_channel_id, -1, 0, 0, 0, capacity);
  return ::rund::detail::task::ChannelDecision{.status =
                                                   task::Status::success()};
}

::rund::detail::task::ChannelDecision
Scheduler::ReleaseChannel(const std::uint64_t,
                          const std::size_t capacity) noexcept {
  EnsureCurrentCommit();
  ReleaseCommittedChannel(0u, capacity);
  return ::rund::detail::task::ChannelDecision{.status =
                                                   task::Status::success()};
}

void Scheduler::ReleaseCommittedChannel(const std::uint64_t,
                                        const std::size_t capacity) noexcept {
  if (state_->resources.live_channels > 0u) {
    --state_->resources.live_channels;
  }
  const std::uint32_t released_slots =
      capacity > std::numeric_limits<std::uint32_t>::max()
          ? std::numeric_limits<std::uint32_t>::max()
          : static_cast<std::uint32_t>(capacity);
  if (state_->resources.live_channel_buffer_slots >= released_slots) {
    state_->resources.live_channel_buffer_slots -= released_slots;
  } else {
    state_->resources.live_channel_buffer_slots = 0u;
  }
}

::rund::detail::task::ChannelDecision
Scheduler::RecordChannel(const ::rund::detail::task::OperationKind kind,
                         const std::uint64_t channel_id,
                         const std::uint64_t value_count) noexcept {
  (void)TrapLaneOwnedSegmentPrimitive(kind);
  EnsureCurrentCommit();
  RecordCommittedChannel(kind, channel_id, value_count);
  return ::rund::detail::task::ChannelDecision{.status =
                                                   task::Status::success()};
}

::rund::detail::task::ChannelDecision Scheduler::RecordBufferedChannelSendBatch(
    const std::uint64_t channel_id, const std::uint64_t value_count,
    const std::uint64_t logical_sends) noexcept {
  if (logical_sends == 0u) {
    return ::rund::detail::task::ChannelDecision{.status =
                                                     task::Status::success()};
  }
  (void)TrapLaneOwnedSegmentPrimitive(
      ::rund::detail::task::OperationKind::ChannelSend);
  EnsureCurrentCommit();
  ::rund::detail::task::Stat(state_->evidence.metrics,
                             ::rund::detail::task::StatSlot::ChannelSends) +=
      logical_sends;
  Record(::rund::detail::task::OperationKind::ChannelSend, ReasonCode::Ok,
         CurrentTaskId(), 0u, 0u, channel_id, -1, 0, 0, 0, value_count, 0u, 0u,
         0u, 0u, 0u, 0u, logical_sends);
  return ::rund::detail::task::ChannelDecision{.status =
                                                   task::Status::success()};
}

void Scheduler::RecordCommittedChannel(
    const ::rund::detail::task::OperationKind kind,
    const std::uint64_t channel_id, const std::uint64_t value_count) noexcept {
  if (kind == ::rund::detail::task::OperationKind::ChannelSend) {
    ++::rund::detail::task::Stat(state_->evidence.metrics,
                                 ::rund::detail::task::StatSlot::ChannelSends);
  } else if (kind == ::rund::detail::task::OperationKind::ChannelRecv) {
    ++::rund::detail::task::Stat(state_->evidence.metrics,
                                 ::rund::detail::task::StatSlot::ChannelRecvs);
  } else if (kind == ::rund::detail::task::OperationKind::ChannelClose) {
    ++::rund::detail::task::Stat(state_->evidence.metrics,
                                 ::rund::detail::task::StatSlot::ChannelCloses);
  }
  const std::uint64_t task_id = CurrentTaskId();
  Record(kind, ReasonCode::Ok, task_id, 0u, 0u, channel_id, -1, 0, 0, 0,
         value_count);
}

} // namespace rund::node
