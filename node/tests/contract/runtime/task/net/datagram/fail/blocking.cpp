#include "src/host/net/test/ticket.hpp"
#include "src/host/net/test/socket.hpp"
#include "../local.hpp"
#include <rund/net/address.hpp>
#include <rund/net/datagram.hpp>
#include <rund/net/listener.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/task/api.hpp>

#include <fcntl.h>

bool NetDatagramTryUsesPerCallNonblocking() {
  SocketCloseGuard sender{};
  SocketCloseGuard receiver{};
  SocketCloseGuard blocking{};
  rund::net::LocalResult sender_address{};
  rund::net::LocalResult receiver_address{};
  DATAGRAM_ASSERT(
      OpenBoundUdpPair(sender, receiver, &sender_address, &receiver_address));

  auto opened_blocking = rund::net::open(
      rund::net::OpenOptions{.family = rund::net::Family::IPv4,
                             .transport = rund::net::Transport::Datagram,
                             .nonblocking = false});
  DATAGRAM_ASSERT(opened_blocking.ok());
  blocking = SocketCloseGuard{std::move(opened_blocking.socket)};
  DATAGRAM_ASSERT(
      rund::net::bind(blocking.socket.view(), LoopbackAnyPort()).ok());
  const int flags_before =
      ::fcntl(rund::node::test::net::native(blocking.socket), F_GETFL, 0);
  DATAGRAM_ASSERT(flags_before >= 0);
  DATAGRAM_ASSERT((flags_before & O_NONBLOCK) == 0);

  std::array<std::byte, 8u> buffer{};
  std::array<std::byte, 1u> one_byte{std::byte{'x'}};
  rund::net::datagram::ReceiveResult blocking_recv{};
  rund::net::datagram::SendResult blocking_send{};
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
        const rund::task::Handle task =
            rund::task::spawn("net-datagram-per-call-nonblocking", [&] {
              blocking_recv = rund::net::datagram::receive(
                  rund::node::test::net::ticket(
                      blocking.socket.view(),
                      rund::net::ready::Interest::Readable),
                  std::span<std::byte>{buffer});
              blocking_send = rund::net::datagram::send(
                  rund::node::test::net::ticket(
                      blocking.socket.view(),
                      rund::net::ready::Interest::Writable),
                  std::span<const std::byte>{one_byte},
                  receiver_address.address);
            });
        joined = rund::task::join(task);
      });

  DATAGRAM_ASSERT(report.ok());
  DATAGRAM_ASSERT(joined.ok());
  DATAGRAM_ASSERT(!blocking_recv.ok());
  DATAGRAM_ASSERT(blocking_recv.code() == rund::ReasonCode::IoWouldBlock);
  DATAGRAM_ASSERT(blocking_send.ok());
  DATAGRAM_ASSERT(blocking_send.bytes == 1);
  DATAGRAM_ASSERT(::fcntl(rund::node::test::net::native(blocking.socket),
                          F_GETFL, 0) == flags_before);
  DATAGRAM_ASSERT(report.events().size() == 2u);
  DATAGRAM_ASSERT(report.events()[0].status == rund::host::Status::WouldBlock);
  DATAGRAM_ASSERT(report.events()[1].status == rund::host::Status::Ok);
  return true;
}
