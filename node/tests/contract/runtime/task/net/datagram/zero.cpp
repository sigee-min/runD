#include "local.hpp"

#include <rund/net/io.hpp>
#include <rund/net/listener.hpp>
#include <rund/task/api.hpp>

bool NetDatagramTransfersEmptyPacket() {
  SocketCloseGuard sender{};
  SocketCloseGuard receiver{};
  rund::net::LocalResult sender_address{};
  rund::net::LocalResult receiver_address{};
  DATAGRAM_ASSERT(
      OpenBoundUdpPair(sender, receiver, &sender_address, &receiver_address));

  std::array<std::byte, 1u> out{std::byte{'z'}};
  std::array<std::byte, 1u> in{};
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
                  .ready_queue_capacity = 2u,
                  .reactor_wait_capacity = 2u,
                  .host_event_capacity = 4u,
              },
      },
      [&] {
        auto transfer = [&]() -> rund::task::Task<void> {
          sent = co_await rund::net::datagram::send(
              sender.socket.view(), std::span<const std::byte>{out.data(), 0u},
              receiver_address.address);
          if (sent) {
            received = co_await rund::net::datagram::receive(
                receiver.socket.view(), std::span<std::byte>{in.data(), 0u});
          }
        };
        const rund::task::Handle task =
            rund::task::spawn("net-empty-datagram", transfer());
        joined = rund::task::join(task);
      });

  DATAGRAM_ASSERT(report.ok());
  DATAGRAM_ASSERT(joined.ok());
  DATAGRAM_ASSERT(sent.ok());
  DATAGRAM_ASSERT(sent.bytes == 0);
  DATAGRAM_ASSERT(received.ok());
  DATAGRAM_ASSERT(received.bytes == 0);
  DATAGRAM_ASSERT(sent.peer_hash.value == receiver_address.address_hash.value);
  DATAGRAM_ASSERT(received.peer_hash.value ==
                  sender_address.address_hash.value);
  DATAGRAM_ASSERT(received.peer == sender_address.address);
  DATAGRAM_ASSERT(report.tasks().network().datagram_send_calls() == 1u);
  DATAGRAM_ASSERT(report.tasks().network().datagram_recv_calls() == 1u);
  return true;
}
