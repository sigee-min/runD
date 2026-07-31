#include "local.hpp"
#include "src/host/net/test/socket.hpp"

#include <rund/net/io.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>

#include "test/assert.hpp"

#include <array>
#include <cstddef>
#include <span>

#include <sys/socket.h>

int RunNetReadinessZeroCase() {
  int sockets[2] = {-1, -1};
  TEST_ASSERT(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
  ReadinessSocketCleanup left_cleanup{sockets[0]};
  ReadinessSocketCleanup right_cleanup{sockets[1]};
  rund::net::Socket left = rund::node::test::net::admit(left_cleanup.fd);
  rund::net::Socket right = rund::node::test::net::admit(right_cleanup.fd);
  TEST_ASSERT(rund::net::nonblocking(left.view(), true).ok());
  TEST_ASSERT(rund::net::nonblocking(right.view(), true).ok());

  std::array<std::byte, 1u> out{std::byte{'z'}};
  std::array<std::byte, 1u> in{};
  rund::net::SendResult sent{};
  rund::net::ReceiveResult received{};
  rund::task::Status joined{};
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 2u,
                  .ready_queue_capacity = 2u,
                  .reactor_wait_capacity = 1u,
                  .host_event_capacity = 2u,
              },
      },
      [&] {
        auto transfer = [&]() -> rund::task::Task<void> {
          sent = co_await rund::net::send(
              left.view(), std::span<const std::byte>{out.data(), 0u});
          received = co_await rund::net::receive(
              right.view(), std::span<std::byte>{in.data(), 0u});
        };
        const rund::task::Handle task =
            rund::task::spawn("net-zero-byte-io", transfer());
        joined = rund::task::join(task);
      });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(sent.ok());
  TEST_ASSERT(sent.bytes == 0);
  TEST_ASSERT(received.ok());
  TEST_ASSERT(received.bytes == 0);
  TEST_ASSERT(report.tasks().reactor_waits() == 0u);
  TEST_ASSERT(report.tasks().network().send_calls() == 1u);
  TEST_ASSERT(report.tasks().network().recv_calls() == 1u);
  TEST_ASSERT(report.events().size() == 2u);
  TEST_ASSERT(report.events()[0].kind == rund::host::EventKind::NetSend);
  TEST_ASSERT(report.events()[1].kind == rund::host::EventKind::NetRecv);
  return 0;
}
