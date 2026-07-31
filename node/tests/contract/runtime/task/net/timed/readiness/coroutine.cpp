#include "src/host/net/test/socket.hpp"
#include "local.hpp"
#include <rund/net/ready/ticket.hpp>
#include <rund/net/ready/timed.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>

#include "test/assert.hpp"

#include <chrono>

int RunTimedReadinessCoroutineCase() {
  TimedSocketPairCleanup coroutine_cleanup{};
  TEST_ASSERT(MakeTimedSocketPair(coroutine_cleanup));
  rund::net::Socket coroutine_reader =
      rund::node::test::net::admit(coroutine_cleanup.left);
  TEST_ASSERT(rund::net::nonblocking(coroutine_reader.view(), true).ok());
  rund::net::ready::Ticket coroutine_result{};
  rund::task::Status coroutine_joined{};
  const rund::Session::Result coroutine_report =
      rund::run(NetTimedReadinessRunSpec(), [&] {
        const auto coroutine_waiter = [&]() -> rund::task::Task<void> {
          coroutine_result = co_await rund::net::ready::timed::read(
              coroutine_reader.view(), std::chrono::milliseconds{1});
          co_return;
        };
        const rund::task::Handle waiter = rund::task::spawn(
            "net-readable-coroutine-timeout", coroutine_waiter());
        coroutine_joined = rund::task::join(waiter);
      });

  TEST_ASSERT(coroutine_report.ok());
  TEST_ASSERT(coroutine_joined.ok());
  TEST_ASSERT(coroutine_result.ok());
  TEST_ASSERT(!coroutine_result.ready());
  TEST_ASSERT(coroutine_result.timed_out());
  TEST_ASSERT(coroutine_result.code() == rund::ReasonCode::IoTimedOut);
  TEST_ASSERT(coroutine_report.tasks().coroutine_parks() >= 1u);
  TEST_ASSERT(coroutine_report.tasks().coroutine_wakes() >= 1u);
  TEST_ASSERT(coroutine_report.tasks().timers() >= 1u);
  TEST_ASSERT(coroutine_report.tasks().reactor_waits() >= 1u);

  TimedSocketPairCleanup coroutine_zero_cleanup{};
  TEST_ASSERT(MakeTimedSocketPair(coroutine_zero_cleanup));
  rund::net::Socket coroutine_zero_reader =
      rund::node::test::net::admit(coroutine_zero_cleanup.left);
  TEST_ASSERT(rund::net::nonblocking(coroutine_zero_reader.view(), true).ok());
  rund::net::ready::Ticket coroutine_zero_result{};
  rund::task::Status coroutine_zero_joined{};
  const rund::Session::Result coroutine_zero_report =
      rund::run(NetTimedReadinessRunSpec(), [&] {
        const auto coroutine_zero_waiter = [&]() -> rund::task::Task<void> {
          coroutine_zero_result = co_await rund::net::ready::timed::read(
              coroutine_zero_reader.view(), std::chrono::nanoseconds{0});
          co_return;
        };
        const rund::task::Handle waiter = rund::task::spawn(
            "net-readable-coroutine-zero-timeout", coroutine_zero_waiter());
        coroutine_zero_joined = rund::task::join(waiter);
      });

  TEST_ASSERT(coroutine_zero_report.ok());
  TEST_ASSERT(coroutine_zero_joined.ok());
  TEST_ASSERT(coroutine_zero_result.ok());
  TEST_ASSERT(!coroutine_zero_result.ready());
  TEST_ASSERT(coroutine_zero_result.timed_out());
  TEST_ASSERT(coroutine_zero_result.code() == rund::ReasonCode::IoTimedOut);
  return 0;
}
