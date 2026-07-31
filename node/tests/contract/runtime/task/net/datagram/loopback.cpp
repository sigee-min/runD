#include "local.hpp"
#include <rund/net/io.hpp>
#include <rund/net/listener.hpp>
#include <rund/task/api.hpp>

bool NetDatagramSendReceiveLoopback() {
  SocketCloseGuard sender{};
  SocketCloseGuard receiver{};
  rund::net::LocalResult sender_address{};
  rund::net::LocalResult receiver_address{};
  DATAGRAM_ASSERT(
      OpenBoundUdpPair(sender, receiver, &sender_address, &receiver_address));

  std::array<std::byte, 7u> payload{
      std::byte{'d'}, std::byte{'a'}, std::byte{'t'}, std::byte{'a'},
      std::byte{'g'}, std::byte{'r'}, std::byte{'m'}};
  std::array<std::byte, payload.size()> received_bytes{};
  rund::net::datagram::SendResult sent{};
  rund::net::datagram::ReceiveResult received{};
  rund::task::Status joined{};

  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_workers = 1u,
                  .task_capacity = 2u,
                  .ready_queue_capacity = 4u,
                  .timer_capacity = 4u,
                  .reactor_wait_capacity = 4u,
                  .host_event_capacity = 8u,
              },
      },
      [&] {
        auto exchange = [&]() -> rund::task::Task<void> {
          sent = co_await rund::net::datagram::send(
              sender.socket.view(), std::span<const std::byte>{payload},
              receiver_address.address);
          if (sent) {
            received = co_await rund::net::datagram::receive(
                receiver.socket.view(), std::span<std::byte>{received_bytes});
          }
        };
        const rund::task::Handle task =
            rund::task::spawn("net-datagram-loopback", exchange());
        joined = rund::task::join(task);
      });
  DATAGRAM_ASSERT(report.ok());
  DATAGRAM_ASSERT(joined.ok());
  DATAGRAM_ASSERT(sent.ok());
  DATAGRAM_ASSERT(sent.bytes == static_cast<std::int64_t>(payload.size()));
  DATAGRAM_ASSERT(sent.peer_hash.value == receiver_address.address_hash.value);
  DATAGRAM_ASSERT(received.ok());
  DATAGRAM_ASSERT(received.bytes == static_cast<std::int64_t>(payload.size()));
  DATAGRAM_ASSERT(received_bytes == payload);
  DATAGRAM_ASSERT(received.peer_hash.value ==
                  sender_address.address_hash.value);
  DATAGRAM_ASSERT(received.peer == sender_address.address);
  DATAGRAM_ASSERT(report.tasks().network().datagram_send_calls() == 1u);
  DATAGRAM_ASSERT(report.tasks().network().datagram_recv_calls() == 1u);
  DATAGRAM_ASSERT(report.tasks().network().bytes_sent() == payload.size());
  DATAGRAM_ASSERT(report.tasks().network().bytes_received() == payload.size());
  return true;
}
