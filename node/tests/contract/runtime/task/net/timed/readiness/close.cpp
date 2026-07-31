#include "src/host/net/test/socket.hpp"
#include "local.hpp"
#include <rund/net/ready/ticket.hpp>
#include <rund/net/ready/timed.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include "test/assert.hpp"

#include <chrono>
#include <cstdint>

int RunTimedReadinessCloseCase() {
  TimedSocketPairCleanup close_cleanup{};
  TEST_ASSERT(MakeTimedSocketPair(close_cleanup));
  rund::net::Socket close_reader =
      rund::node::test::net::admit(close_cleanup.left);
  rund::net::Socket close_writer =
      rund::node::test::net::admit(close_cleanup.right);
  TEST_ASSERT(rund::net::nonblocking(close_reader.view(), true).ok());
  TEST_ASSERT(rund::net::nonblocking(close_writer.view(), true).ok());

  rund::net::ready::Ticket close_wait_result{};
  rund::net::CloseResult close_result{};
  rund::task::Status close_yielded{};
  rund::task::Status sleep_yielded{};
  rund::task::Status post_close_sleep{};
  rund::task::Status close_joined{};
  const rund::Session::Result close_report =
      rund::run(NetTimedReadinessRunSpec(), [&] {
        auto wait = [&]() -> rund::task::Task<void> {
          close_wait_result = co_await rund::net::ready::timed::read(
              close_reader.view(), std::chrono::milliseconds{50});
        };
        const rund::task::Handle waiter =
            rund::task::spawn("net-readable-timeout-close", wait());
        auto close = [&]() -> rund::task::Task<void> {
          close_yielded = co_await rund::task::yield();
          close_result = close_reader.close();
        };
        const rund::task::Handle closer =
            rund::task::spawn("net-close-timed-wait", close());
        auto sleep = [&]() -> rund::task::Task<void> {
          sleep_yielded = co_await rund::task::yield();
          post_close_sleep =
              co_await rund::task::sleep(std::chrono::milliseconds{75});
        };
        const rund::task::Handle sleeper =
            rund::task::spawn("net-close-timed-wait-sleeper", sleep());
        close_joined = rund::task::join(waiter, closer, sleeper);
      });

  std::uint64_t close_timer_sleep_events = 0u;
  for (const rund::host::Event &event : close_report.events()) {
    if (event.kind == rund::host::EventKind::TimerSleep) {
      ++close_timer_sleep_events;
    }
  }

  TEST_ASSERT(close_report.ok());
  TEST_ASSERT(close_joined.ok());
  TEST_ASSERT(close_yielded.ok());
  TEST_ASSERT(sleep_yielded.ok());
  TEST_ASSERT(post_close_sleep.ok());
  TEST_ASSERT(close_result.ok());
  TEST_ASSERT(!close_wait_result.ok());
  TEST_ASSERT(close_wait_result.code() == rund::ReasonCode::IoFdInvalid);
  TEST_ASSERT(close_timer_sleep_events == 1u);
  return 0;
}
