#include "local.hpp"
#include <rund/net/io.hpp>
#include <rund/net/listener.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include <rund/replay.hpp>

bool NetDatagramReplayEventsAreStable() {
  SocketCloseGuard sender{};
  SocketCloseGuard receiver{};
  rund::net::LocalResult sender_address{};
  rund::net::LocalResult receiver_address{};
  DATAGRAM_ASSERT(
      OpenBoundUdpPair(sender, receiver, &sender_address, &receiver_address));
  const std::array<std::byte, 4u> payload{std::byte{'r'}, std::byte{'e'},
                                          std::byte{'p'}, std::byte{'l'}};
  std::array<std::byte, 4u> received_bytes{};
  rund::net::datagram::SendResult sent{};
  rund::net::datagram::ReceiveResult received{};
  rund::task::Status joined{};
  rund::task::Status sender_joined{};
  auto exchange = [&](rund::replay::Context &) {
    auto task = [&]() -> rund::task::Task<void> {
      auto send = [&]() -> rund::task::Task<void> {
        sent = co_await rund::net::datagram::send(
            sender.socket.view(), std::span<const std::byte>{payload},
            receiver_address.address);
      };
      const rund::task::Handle sender_task =
          rund::task::spawn("net-datagram-replay-send", send());

      // Register the read before the sender can run. This removes the
      // scheduler-versus-kernel arrival race from the contract: both Record
      // and Replay take the same parked readiness path instead of sometimes
      // observing an already queued datagram.
      received = co_await rund::net::datagram::receive(
          receiver.socket.view(), std::span<std::byte>{received_bytes});
      sender_joined = co_await sender_task;
    };
    const rund::task::Handle handle =
        rund::task::spawn("net-datagram-replay", task());
    joined = rund::task::join(handle);
  };

  rund::Session session{};
  DATAGRAM_ASSERT(session.open(rund::SessionConfig{
      .workers = 1u,
      .scheduler =
          {
              .task_workers = 1u,
              .task_capacity = 2u,
              .ready_queue_capacity = 4u,
              .timer_capacity = 4u,
              .reactor_wait_capacity = 4u,
              .host_event_capacity = 8u,
              .host_payload_capacity_bytes = payload.size() * 2u,
          },
  }));
  const rund::replay::Record recorded = rund::replay::record(session, exchange);
  DATAGRAM_ASSERT(recorded.ok());
  DATAGRAM_ASSERT(joined.ok());
  DATAGRAM_ASSERT(sender_joined.ok());
  DATAGRAM_ASSERT(sent.ok());
  DATAGRAM_ASSERT(received.ok());
  DATAGRAM_ASSERT(received_bytes == payload);
  DATAGRAM_ASSERT(recorded.host_event_count() >= 3u);
  DATAGRAM_ASSERT(recorded.tasks().network().bytes_sent() == payload.size());
  DATAGRAM_ASSERT(recorded.tasks().network().bytes_received() ==
                  payload.size());

  received_bytes.fill(std::byte{});
  const rund::replay::Check replayed =
      rund::replay::run(session, recorded, exchange);
  DATAGRAM_ASSERT(replayed.ok());
  DATAGRAM_ASSERT(joined.ok());
  DATAGRAM_ASSERT(sender_joined.ok());
  DATAGRAM_ASSERT(sent.ok());
  DATAGRAM_ASSERT(received.ok());
  DATAGRAM_ASSERT(received_bytes == payload);
  DATAGRAM_ASSERT(replayed.actual().has_value());
  DATAGRAM_ASSERT(replayed.actual()->host_event_hash() ==
                  recorded.host_event_hash());
  DATAGRAM_ASSERT(replayed.actual()->tasks().network().bytes_sent() ==
                  recorded.tasks().network().bytes_sent());
  DATAGRAM_ASSERT(replayed.actual()->tasks().network().bytes_received() ==
                  recorded.tasks().network().bytes_received());
  DATAGRAM_ASSERT(session.close());
  return true;
}
