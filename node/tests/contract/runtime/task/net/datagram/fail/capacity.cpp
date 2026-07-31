#include "../local.hpp"
#include <rund/net/io.hpp>
#include <rund/net/listener.hpp>
#include <rund/task/api.hpp>

bool NetDatagramRejectsRuntimeCapacityOversize() {
  SocketCloseGuard sender{};
  SocketCloseGuard receiver{};
  rund::net::LocalResult sender_address{};
  rund::net::LocalResult receiver_address{};
  DATAGRAM_ASSERT(
      OpenBoundUdpPair(sender, receiver, &sender_address, &receiver_address));

  std::vector<std::byte> oversized(9u, std::byte{'o'});
  rund::net::datagram::ReceiveResult oversized_recv{};
  rund::net::datagram::SendResult oversized_send{};
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
                  .net_datagram_capacity_bytes = 8u,
                  .host_event_capacity = 8u,
              },
      },
      [&] {
        auto reject = [&]() -> rund::task::Task<void> {
          oversized_recv = co_await rund::net::datagram::receive(
              receiver.socket.view(),
              std::span<std::byte>{oversized.data(), oversized.size()});
          oversized_send = co_await rund::net::datagram::send(
              sender.socket.view(),
              std::span<const std::byte>{oversized.data(), oversized.size()},
              receiver_address.address);
        };
        const rund::task::Handle task =
            rund::task::spawn("net-datagram-capacity-failures", reject());
        joined = rund::task::join(task);
      });

  DATAGRAM_ASSERT(report.ok());
  DATAGRAM_ASSERT(joined.ok());
  DATAGRAM_ASSERT(!oversized_recv.ok());
  DATAGRAM_ASSERT(oversized_recv.code() == rund::ReasonCode::TaskInvalid);
  DATAGRAM_ASSERT(!oversized_send.ok());
  DATAGRAM_ASSERT(oversized_send.code() == rund::ReasonCode::TaskInvalid);
  DATAGRAM_ASSERT(report.tasks().reactor_waits() == 0u);
  DATAGRAM_ASSERT(report.tasks().network().datagram_recv_calls() == 0u);
  DATAGRAM_ASSERT(report.tasks().network().datagram_send_calls() == 0u);
  DATAGRAM_ASSERT(report.events().empty());
  return true;
}
