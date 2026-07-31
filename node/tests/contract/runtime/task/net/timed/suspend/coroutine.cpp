#include "src/host/net/test/socket.hpp"
#include "local.hpp"
#include <rund/net/bytes.hpp>
#include <rund/net/ready.hpp>
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
#include <utility>

namespace {

[[nodiscard]] int RunTimedSuspendCoroutineReadyCase() {
  TimedSuspendSocketPairCleanup cleanup{};
  TEST_ASSERT(MakeTimedSuspendSocketPair(cleanup));
  rund::net::Socket reader = rund::node::test::net::admit(cleanup.left);
  rund::net::Socket writer = rund::node::test::net::admit(cleanup.right);
  TEST_ASSERT(rund::net::nonblocking(reader.view(), true).ok());
  TEST_ASSERT(rund::net::nonblocking(writer.view(), true).ok());

  std::array<std::byte, 1u> out{std::byte{'r'}};
  rund::net::ready::Ticket ready_result{};
  rund::net::SendResult sent{};
  rund::task::Status writer_yielded{};
  rund::task::Status joined{};
  const rund::Session::Result report =
      rund::run(TimedReadySuspendRunSpec(), [&] {
        const auto waiter = [&]() -> rund::task::Task<void> {
          auto operation = rund::net::ready::timed::read(
              reader.view(), std::chrono::milliseconds{100});
          auto owner = std::move(operation);
          ready_result = co_await std::move(owner);
          co_return;
        };
        const rund::task::Handle reader_task =
            rund::task::spawn("net-timed-ready-coroutine-positive", waiter());
        const auto write = [&]() -> rund::task::Task<void> {
          writer_yielded = co_await rund::task::yield();
          rund::net::ready::Ticket writable =
              co_await rund::net::ready::write(writer.view());
          sent = rund::net::send(std::move(writable),
                                 std::span<const std::byte>{out});
        };
        const rund::task::Handle writer_task =
            rund::task::spawn("net-timed-ready-coroutine-writer", write());
        joined = rund::task::join(reader_task, writer_task);
      });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(sent.ok());
  TEST_ASSERT(writer_yielded.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(ready_result.ok());
  TEST_ASSERT(ready_result.ready());
  TEST_ASSERT(!ready_result.timed_out());
  TEST_ASSERT(report.tasks().coroutine_parks() >= 1u);
  TEST_ASSERT(report.tasks().coroutine_wakes() >= 1u);
  return 0;
}

} // namespace

int RunTimedSuspendCoroutineCase() {
  TEST_ASSERT(RunTimedSuspendCoroutineReadyCase() == 0);
  return 0;
}
