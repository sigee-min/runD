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
#include <unistd.h>

int RunNetReadinessHupCase() {
  int hup_sockets[2] = {-1, -1};
  TEST_ASSERT(MakeReadinessSocketPair(hup_sockets));
  ReadinessSocketCleanup hup_left_cleanup{hup_sockets[0]};
  ReadinessSocketCleanup hup_right_cleanup{hup_sockets[1]};
  rund::net::Socket hup_right =
      rund::node::test::net::admit(hup_right_cleanup.fd);
  TEST_ASSERT(rund::net::nonblocking(hup_right.view(), true).ok());
  std::array<std::byte, 1u> hup_bytes{};
  rund::net::ReceiveResult hup_received{};
  rund::task::Status join_result{};
  const rund::Session::Result hup_report = rund::run(
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
          hup_received = co_await rund::net::receive(
              hup_right.view(), std::span<std::byte>{hup_bytes});
        };
        const rund::task::Handle reader =
            rund::task::spawn("net-hup-readable", read());
        const rund::task::Handle closer =
            rund::task::spawn("net-hup-close-peer", [&] {
              static_cast<void>(::close(hup_left_cleanup.fd));
              hup_left_cleanup.fd = -1;
            });
        join_result = rund::task::join(reader, closer);
      });
  TEST_ASSERT(hup_report.ok());
  TEST_ASSERT(join_result.ok());
  TEST_ASSERT(hup_received.ok());
  TEST_ASSERT(hup_received.bytes == 0);
  TEST_ASSERT(hup_report.tasks().reactor_waits() == 1u);
  TEST_ASSERT(hup_report.tasks().network().recv_calls() == 1u);
  TEST_ASSERT(hup_report.events().size() == 2u);
  TEST_ASSERT(hup_report.events()[0].kind == rund::host::EventKind::IoReady);
  TEST_ASSERT(hup_report.events()[1].kind == rund::host::EventKind::NetRecv);
  TEST_ASSERT(hup_report.events()[0].host_handle_id != 0u);
  TEST_ASSERT(hup_report.events()[0].host_handle_id ==
              hup_report.events()[1].host_handle_id);
  return 0;
}
