#include "src/host/net/test/ticket.hpp"
#include "src/host/net/test/socket.hpp"
#include "local.hpp"
#include <rund/net/bytes.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/ready/timed.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include "test/assert.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <span>

int RunTimedReadinessReadyCase() {
  TimedSocketPairCleanup ready_cleanup{};
  TEST_ASSERT(MakeTimedSocketPair(ready_cleanup));
  rund::net::Socket ready_reader =
      rund::node::test::net::admit(ready_cleanup.left);
  rund::net::Socket ready_writer =
      rund::node::test::net::admit(ready_cleanup.right);
  TEST_ASSERT(rund::net::nonblocking(ready_reader.view(), true).ok());
  TEST_ASSERT(rund::net::nonblocking(ready_writer.view(), true).ok());

  std::array<std::byte, 1u> out{std::byte{'t'}};
  rund::net::ready::Ticket ready_result{};
  rund::net::SendResult sent{};
  rund::task::Status yielded{};
  rund::task::Status ready_joined{};
  const rund::Session::Result ready_report =
      rund::run(NetTimedReadinessRunSpec(), [&] {
        auto wait = [&]() -> rund::task::Task<void> {
          ready_result = co_await rund::net::ready::timed::read(
              ready_reader.view(), std::chrono::milliseconds{100});
        };
        const rund::task::Handle waiter =
            rund::task::spawn("net-readable-before-timeout", wait());
        auto write = [&]() -> rund::task::Task<void> {
          yielded = co_await rund::task::yield();
          sent = rund::net::send(
              rund::node::test::net::ticket(
                  ready_writer.view(), rund::net::ready::Interest::Writable),
              std::span<const std::byte>{out});
        };
        const rund::task::Handle writer =
            rund::task::spawn("net-readable-timeout-writer", write());
        ready_joined = rund::task::join(waiter, writer);
      });

  TEST_ASSERT(ready_report.ok());
  TEST_ASSERT(ready_joined.ok());
  TEST_ASSERT(yielded.ok());
  TEST_ASSERT(sent.ok());
  TEST_ASSERT(ready_result.ok());
  TEST_ASSERT(ready_result.ready());
  TEST_ASSERT(!ready_result.timed_out());
  return 0;
}
