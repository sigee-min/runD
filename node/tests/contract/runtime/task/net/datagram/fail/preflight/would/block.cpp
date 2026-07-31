#include "src/host/net/test/ticket.hpp"
#include "../local.hpp"
#include <rund/net/datagram.hpp>
#include <rund/net/listener.hpp>
#include <rund/net/ready/ticket.hpp>

bool NetDatagramRejectsWouldBlockPreflight() {
  SocketCloseGuard sender{};
  SocketCloseGuard receiver{};
  rund::net::LocalResult sender_address{};
  rund::net::LocalResult receiver_address{};
  DATAGRAM_ASSERT(
      OpenBoundUdpPair(sender, receiver, &sender_address, &receiver_address));

  std::array<std::byte, 8u> buffer{};
  const rund::net::datagram::ReceiveResult no_ready = rund::net::datagram::receive(
      rund::node::test::net::ticket(receiver.socket.view(),
                                    rund::net::ready::Interest::Readable),
      std::span<std::byte>{buffer});
  DATAGRAM_ASSERT(!no_ready.ok());
  DATAGRAM_ASSERT(no_ready.code() == rund::ReasonCode::IoWouldBlock);
  return true;
}
