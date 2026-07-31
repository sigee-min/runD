#include "src/host/net/test/socket.hpp"
#include "local.hpp"
#include <rund/net/ready.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>

#include "test/assert.hpp"

int RunNetReadinessWritableCase() {
  int sockets[2] = {-1, -1};
  TEST_ASSERT(MakeReadinessSocketPair(sockets));
  ReadinessSocketCleanup left_cleanup{sockets[0]};
  ReadinessSocketCleanup right_cleanup{sockets[1]};
  rund::net::Socket left = rund::node::test::net::admit(left_cleanup.fd);
  TEST_ASSERT(rund::net::nonblocking(left.view(), true).ok());

  rund::net::ready::Ticket writable_ready{};
  rund::task::Status join_result{};
  const rund::Session::Result writable_report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 2u,
                  .ready_queue_capacity = 2u,
                  .host_event_capacity = 2u,
              },
      },
      [&] {
        auto wait = [&]() -> rund::task::Task<void> {
          writable_ready = co_await rund::net::ready::write(left.view());
        };
        const rund::task::Handle task =
            rund::task::spawn("net-writable", wait());
        join_result = rund::task::join(task);
      });
  TEST_ASSERT(writable_report.ok());
  TEST_ASSERT(join_result.ok());
  TEST_ASSERT(writable_ready.ok());
  TEST_ASSERT(writable_report.events().size() == 1u);
  TEST_ASSERT(writable_report.events()[0].kind ==
              rund::host::EventKind::IoReady);
  TEST_ASSERT(writable_report.events()[0].host_handle_id != 0u);
  return 0;
}
