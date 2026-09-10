#include "local.hpp"

namespace rund::node {

void record_detail::RecordNetworkStats(
    ::rund::detail::task::StatStorage &stats,
    const ::rund::host::Event &event) noexcept {
  ::rund::detail::task::StatSlot call_slot =
      ::rund::detail::task::StatSlot::Count;
  ::rund::detail::task::StatSlot byte_slot =
      ::rund::detail::task::StatSlot::Count;
  switch (event.kind) {
  case ::rund::host::EventKind::NetSocket:
    call_slot = ::rund::detail::task::StatSlot::NetworkSocketsOpened;
    break;
  case ::rund::host::EventKind::NetBind:
    call_slot = ::rund::detail::task::StatSlot::NetworkSocketsBound;
    break;
  case ::rund::host::EventKind::NetListen:
    call_slot = ::rund::detail::task::StatSlot::NetworkSocketsListened;
    break;
  case ::rund::host::EventKind::NetShutdown:
    call_slot = ::rund::detail::task::StatSlot::NetworkSocketsShutdown;
    break;
  case ::rund::host::EventKind::NetLocalAddress:
    call_slot = ::rund::detail::task::StatSlot::NetworkLocalAddressReads;
    break;
  case ::rund::host::EventKind::NetAccept:
    call_slot = ::rund::detail::task::StatSlot::NetworkAccepts;
    break;
  case ::rund::host::EventKind::NetConnect:
    call_slot = ::rund::detail::task::StatSlot::NetworkConnects;
    break;
  case ::rund::host::EventKind::NetRecv:
    call_slot = ::rund::detail::task::StatSlot::NetworkRecvCalls;
    byte_slot = ::rund::detail::task::StatSlot::NetworkBytesReceived;
    break;
  case ::rund::host::EventKind::NetSend:
    call_slot = ::rund::detail::task::StatSlot::NetworkSendCalls;
    byte_slot = ::rund::detail::task::StatSlot::NetworkBytesSent;
    break;
  case ::rund::host::EventKind::NetRecvDatagram:
    call_slot = ::rund::detail::task::StatSlot::NetworkDatagramRecvCalls;
    byte_slot = ::rund::detail::task::StatSlot::NetworkBytesReceived;
    break;
  case ::rund::host::EventKind::NetSendDatagram:
    call_slot = ::rund::detail::task::StatSlot::NetworkDatagramSendCalls;
    byte_slot = ::rund::detail::task::StatSlot::NetworkBytesSent;
    break;
  case ::rund::host::EventKind::NetSetSocketOption:
    call_slot = ::rund::detail::task::StatSlot::NetworkSocketOptionsSet;
    break;
  case ::rund::host::EventKind::NetGetSocketOption:
    call_slot = ::rund::detail::task::StatSlot::NetworkSocketOptionsRead;
    break;
  case ::rund::host::EventKind::NetRecvVectored:
    call_slot = ::rund::detail::task::StatSlot::NetworkVectoredRecvCalls;
    byte_slot = ::rund::detail::task::StatSlot::NetworkBytesReceived;
    break;
  case ::rund::host::EventKind::NetSendVectored:
    call_slot = ::rund::detail::task::StatSlot::NetworkVectoredSendCalls;
    byte_slot = ::rund::detail::task::StatSlot::NetworkBytesSent;
    break;
  default:
    break;
  }
  if (call_slot == ::rund::detail::task::StatSlot::Count) {
    return;
  }
  if (event.status == ::rund::host::Status::WouldBlock) {
    ::rund::detail::counter::Accumulate(
        ::rund::detail::task::Stat(
            stats, ::rund::detail::task::StatSlot::NetworkWouldBlock),
        1u);
    return;
  }
  if (event.status != ::rund::host::Status::Ok) {
    return;
  }
  ::rund::detail::counter::Accumulate(
      ::rund::detail::task::Stat(stats, call_slot), 1u);
  if (byte_slot != ::rund::detail::task::StatSlot::Count) {
    ::rund::detail::counter::Accumulate(
        ::rund::detail::task::Stat(stats, byte_slot), event.completed_bytes);
  }
}

} // namespace rund::node
