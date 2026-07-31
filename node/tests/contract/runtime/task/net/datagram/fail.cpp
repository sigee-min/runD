#include "src/host/net/test/ticket.hpp"
#include "local.hpp"
#include <rund/net/datagram.hpp>
#include <rund/net/listener.hpp>
#include <rund/net/ready/ticket.hpp>

bool NetDatagramWouldBlockAndInvalidInputsFailClosed() {
  DATAGRAM_ASSERT(NetDatagramPreflightFailuresFailClosed());
  DATAGRAM_ASSERT(NetDatagramTryUsesPerCallNonblocking());
  DATAGRAM_ASSERT(NetDatagramRejectsRuntimeCapacityOversize());
  return true;
}

bool NetDatagramRejectsOversizedRequest() {
  SocketCloseGuard sender{};
  SocketCloseGuard receiver{};
  rund::net::LocalResult sender_address{};
  rund::net::LocalResult receiver_address{};
  DATAGRAM_ASSERT(
      OpenBoundUdpPair(sender, receiver, &sender_address, &receiver_address));

  std::vector<std::byte> oversized(
      rund::SchedulerConfig{}.net_datagram_capacity_bytes + 1u);
  auto ticket = rund::node::test::net::ticket(
      sender.socket.view(), rund::net::ready::Interest::Writable);
  const rund::net::datagram::SendResult sent = rund::net::datagram::send(
      std::move(ticket),
      std::span<const std::byte>{oversized.data(), oversized.size()},
      receiver_address.address);
  DATAGRAM_ASSERT(!sent.ok());
  DATAGRAM_ASSERT(sent.code() == rund::ReasonCode::TaskInvalid);
  DATAGRAM_ASSERT(ticket.consumed());
  const rund::net::datagram::SendResult reused = rund::net::datagram::send(
      std::move(ticket), {}, receiver_address.address);
  DATAGRAM_ASSERT(!reused.ok());
  DATAGRAM_ASSERT(reused.code() == rund::ReasonCode::NetTicketConsumed);
  return true;
}
