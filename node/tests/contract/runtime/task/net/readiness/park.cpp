#include "local.hpp"
#include "src/host/net/test/socket.hpp"
#include <rund/net/io.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>

#include "test/assert.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <utility>

int RunNetReadinessParkedCase() {
  int parked_sockets[2] = {-1, -1};
  TEST_ASSERT(MakeReadinessSocketPair(parked_sockets));
  ReadinessSocketCleanup parked_left_cleanup{parked_sockets[0]};
  ReadinessSocketCleanup parked_right_cleanup{parked_sockets[1]};
  rund::net::Socket parked_left =
      rund::node::test::net::admit(parked_left_cleanup.fd);
  rund::net::Socket parked_right =
      rund::node::test::net::admit(parked_right_cleanup.fd);
  TEST_ASSERT(rund::net::nonblocking(parked_left.view(), true).ok());
  TEST_ASSERT(rund::net::nonblocking(parked_right.view(), true).ok());
  std::array<std::byte, 1> parked_out{std::byte{'p'}};
  std::array<std::byte, 1> parked_in{};
  rund::net::ReceiveResult parked_recv{};
  rund::net::SendResult parked_send{};
  rund::task::Status join_result{};
  const rund::Session::Result parked_report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 4u,
                  .ready_queue_capacity = 4u,
                  .reactor_wait_capacity = 2u,
                  .host_event_capacity = 4u,
              },
      },
      [&] {
        auto read = [&]() -> rund::task::Task<void> {
          parked_recv = co_await rund::net::receive(
              parked_right.view(), std::span<std::byte>{parked_in});
        };
        const rund::task::Handle reader =
            rund::task::spawn("net-parked-readable", read());
        auto write = [&]() -> rund::task::Task<void> {
          parked_send = co_await rund::net::send(
              parked_left.view(), std::span<const std::byte>{parked_out});
        };
        const rund::task::Handle writer =
            rund::task::spawn("net-parked-writer", write());
        join_result = rund::task::join(reader, writer);
      });
  TEST_ASSERT(parked_report.ok());
  TEST_ASSERT(join_result.ok());
  TEST_ASSERT(parked_send.ok());
  TEST_ASSERT(parked_recv.ok());
  TEST_ASSERT(parked_in == parked_out);
  TEST_ASSERT(parked_report.tasks().reactor_waits() == 1u);
  TEST_ASSERT(parked_report.tasks().network().send_calls() == 1u);
  TEST_ASSERT(parked_report.tasks().network().recv_calls() == 1u);
  TEST_ASSERT(parked_report.events().size() == 4u);
  TEST_ASSERT(parked_report.events()[0].kind == rund::host::EventKind::IoReady);
  TEST_ASSERT(parked_report.events()[1].kind == rund::host::EventKind::NetSend);
  TEST_ASSERT(parked_report.events()[2].kind == rund::host::EventKind::IoReady);
  TEST_ASSERT(parked_report.events()[3].kind == rund::host::EventKind::NetRecv);
  TEST_ASSERT(parked_report.events()[0].host_handle_id != 0u);
  TEST_ASSERT(parked_report.events()[0].host_handle_id ==
              parked_report.events()[1].host_handle_id);
  TEST_ASSERT(parked_report.events()[2].host_handle_id != 0u);
  TEST_ASSERT(parked_report.events()[2].host_handle_id ==
              parked_report.events()[3].host_handle_id);
  return 0;
}
