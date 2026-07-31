#include "../local.hpp"
#include <rund/net/address.hpp>
#include <rund/net/listener.hpp>

[[nodiscard]] bool
OpenBoundUdpPair(SocketCloseGuard &sender, SocketCloseGuard &receiver,
                 rund::net::LocalResult *const sender_address,
                 rund::net::LocalResult *const receiver_address) {
  auto opened_sender = rund::net::open(
      rund::net::OpenOptions{.family = rund::net::Family::IPv4,
                             .transport = rund::net::Transport::Datagram,
                             .nonblocking = true});
  DATAGRAM_ASSERT(opened_sender.ok());
  DATAGRAM_ASSERT(opened_sender.nonblocking.ok());
  sender = SocketCloseGuard{std::move(opened_sender.socket)};

  auto opened_receiver = rund::net::open(
      rund::net::OpenOptions{.family = rund::net::Family::IPv4,
                             .transport = rund::net::Transport::Datagram,
                             .nonblocking = true});
  DATAGRAM_ASSERT(opened_receiver.ok());
  DATAGRAM_ASSERT(opened_receiver.nonblocking.ok());
  receiver = SocketCloseGuard{std::move(opened_receiver.socket)};

  const auto sender_bind =
      rund::net::bind(sender.socket.view(), LoopbackAnyPort());
  DATAGRAM_ASSERT(sender_bind.ok());
  const auto receiver_bind =
      rund::net::bind(receiver.socket.view(), LoopbackAnyPort());
  DATAGRAM_ASSERT(receiver_bind.ok());

  *sender_address = rund::net::local(sender.socket.view());
  *receiver_address = rund::net::local(receiver.socket.view());
  DATAGRAM_ASSERT(sender_address->ok());
  DATAGRAM_ASSERT(receiver_address->ok());
  DATAGRAM_ASSERT(IsLoopbackPort(sender_address->address));
  DATAGRAM_ASSERT(IsLoopbackPort(receiver_address->address));
  DATAGRAM_ASSERT(sender_address->address_hash.value != 0u);
  DATAGRAM_ASSERT(receiver_address->address_hash.value != 0u);
  return true;
}
