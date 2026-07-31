#include <rund/task/stats/slots.hpp>

#include "../../../state/segment.hpp"
#include "../../../state/storage.hpp"

namespace rund::node {

void Scheduler::CommitLaneOwnedPrimitiveTrapEffect(
    LaneOwnedTerminalRange &terminal_range,
    const LaneSegmentEffect &effect) noexcept {
  FlushLaneOwnedTerminalRange(terminal_range);
  Record(::rund::detail::task::OperationKind::PrimitiveTrap, effect.trap_code,
         effect.task_id, 0u, 0u, 0u, -1, 0, 0, 0, 1u, 0u, 0u, 0u, 0u, 0u, 0u,
         1u, static_cast<std::uint64_t>(effect.trap_kind), effect.trap_code);
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::PrimitiveTrapPackets);
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::LaneLocalPrimitiveTrapLogPackets);
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::LaneLocalPrimitiveTrapLogicalEvents);
  if (effect.trap_kind == ::rund::detail::task::OperationKind::ChannelSend) {
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::LaneLocalPrimitiveTrapChannelSends);
  } else if (effect.trap_kind ==
             ::rund::detail::task::OperationKind::ChannelRecv) {
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::LaneLocalPrimitiveTrapChannelRecvs);
  } else if (effect.trap_kind ==
             ::rund::detail::task::OperationKind::ChannelClose) {
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::LaneLocalPrimitiveTrapChannelCloses);
  }
  ++::rund::detail::task::Stat(state_->evidence.metrics,
                               ::rund::detail::task::StatSlot::SideExits);
}

} // namespace rund::node
