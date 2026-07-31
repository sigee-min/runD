#include "src/host/net/test/ticket.hpp"
#include "local.hpp"
#include <rund/net/address.hpp>
#include <rund/net/datagram.hpp>
#include <rund/net/listener.hpp>
#include <rund/net/ready/ticket.hpp>

bool NetDatagramRejectsNullPreflight() {
  SocketCloseGuard sender{};
  SocketCloseGuard receiver{};
  rund::net::LocalResult sender_address{};
  rund::net::LocalResult receiver_address{};
  DATAGRAM_ASSERT(
      OpenBoundUdpPair(sender, receiver, &sender_address, &receiver_address));

  auto receive_ticket = rund::node::test::net::ticket(
      receiver.socket.view(), rund::net::ready::Interest::Readable);
  const rund::net::datagram::ReceiveResult null_recv = rund::net::datagram::receive(
      std::move(receive_ticket),
      std::span<std::byte>{static_cast<std::byte *>(nullptr), 1u});
  DATAGRAM_ASSERT(!null_recv.ok());
  DATAGRAM_ASSERT(null_recv.code() == rund::ReasonCode::TaskInvalid);
  DATAGRAM_ASSERT(receive_ticket.consumed());
  const rund::net::datagram::ReceiveResult reused_recv =
      rund::net::datagram::receive(std::move(receive_ticket), {});
  DATAGRAM_ASSERT(!reused_recv.ok());
  DATAGRAM_ASSERT(reused_recv.code() == rund::ReasonCode::NetTicketConsumed);

  auto send_ticket = rund::node::test::net::ticket(
      sender.socket.view(), rund::net::ready::Interest::Writable);
  const rund::net::datagram::SendResult null_send = rund::net::datagram::send(
      std::move(send_ticket),
      std::span<const std::byte>{static_cast<const std::byte *>(nullptr), 1u},
      receiver_address.address);
  DATAGRAM_ASSERT(!null_send.ok());
  DATAGRAM_ASSERT(null_send.code() == rund::ReasonCode::TaskInvalid);
  DATAGRAM_ASSERT(send_ticket.consumed());
  const rund::net::datagram::SendResult reused_send = rund::net::datagram::send(
      std::move(send_ticket), {}, receiver_address.address);
  DATAGRAM_ASSERT(!reused_send.ok());
  DATAGRAM_ASSERT(reused_send.code() == rund::ReasonCode::NetTicketConsumed);

  auto peer_ticket = rund::node::test::net::ticket(
      sender.socket.view(), rund::net::ready::Interest::Writable);
  const rund::net::datagram::SendResult invalid_peer = rund::net::datagram::send(
      std::move(peer_ticket), {}, rund::net::Address{});
  DATAGRAM_ASSERT(!invalid_peer.ok());
  DATAGRAM_ASSERT(invalid_peer.code() == rund::ReasonCode::TaskInvalid);
  DATAGRAM_ASSERT(peer_ticket.consumed());
  const rund::net::datagram::SendResult reused_peer = rund::net::datagram::send(
      std::move(peer_ticket), {}, receiver_address.address);
  DATAGRAM_ASSERT(!reused_peer.ok());
  DATAGRAM_ASSERT(reused_peer.code() == rund::ReasonCode::NetTicketConsumed);
  return true;
}
