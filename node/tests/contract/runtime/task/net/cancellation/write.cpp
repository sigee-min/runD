#include "src/host/net/test/socket.hpp"
#include "local.hpp"
#include <rund/net/cancel.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/ready/timed.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>
#include <rund/task/cancel.hpp>

#include "test/assert.hpp"

#include <chrono>

int RunNetCancellationWritableCase() {
  CancelSocketPairCleanup writable_cleanup{};
  TEST_ASSERT(MakeCancelSocketPair(writable_cleanup));
  rund::net::Socket writable_reader =
      rund::node::test::net::admit(writable_cleanup.left);
  rund::net::Socket writable_writer =
      rund::node::test::net::admit(writable_cleanup.right);
  TEST_ASSERT(rund::net::nonblocking(writable_reader.view(), true).ok());
  TEST_ASSERT(rund::net::nonblocking(writable_writer.view(), true).ok());
  TEST_ASSERT(SaturateSocket(rund::node::test::net::native(writable_writer)));

  rund::net::ready::Ticket writable_ready_result{};
  bool writable_cancel_ok = false;
  rund::task::Status writable_cancel_yield{};
  rund::net::CloseResult writable_close_result{};
  rund::task::Status writable_scope{};
  rund::task::Status writable_post_close_scope{};
  rund::task::Status writable_post_close_sleep{};
  bool writable_source_valid = false;
  bool writable_token_valid = false;
  const rund::Session::Result writable_report =
      rund::run(NetCancellationRunSpec(), [&] {
        auto source = rund::task::stop_source::create();
        writable_source_valid = static_cast<bool>(source);
        if (!writable_source_valid) {
          return;
        }
        auto token = source.token();
        writable_token_valid = static_cast<bool>(token);
        if (!writable_token_valid) {
          return;
        }

        auto wait = [&]() -> rund::task::Task<void> {
          writable_ready_result = co_await rund::net::ready::timed::write(
              writable_writer.view(), std::chrono::seconds{30}, token);
        };
        auto cancel = [&]() -> rund::task::Task<void> {
          writable_cancel_yield = co_await rund::task::yield();
          writable_cancel_ok = source.request_stop().ok();
        };
        writable_scope = rund::task::scope([&] {
          (void)rund::task::spawn("net-cancel-writable-waiter", wait());
          (void)rund::task::spawn("net-cancel-writable-requester", cancel());
        });

        writable_close_result = writable_writer.close();

        auto sleep = [&]() -> rund::task::Task<void> {
          writable_post_close_sleep =
              co_await rund::task::sleep(std::chrono::milliseconds{1});
        };
        writable_post_close_scope = rund::task::scope([&] {
          (void)rund::task::spawn("net-cancel-writable-short-scope", sleep());
        });
      });

  TEST_ASSERT(writable_report.ok());
  TEST_ASSERT(writable_source_valid);
  TEST_ASSERT(writable_token_valid);
  TEST_ASSERT(writable_scope.ok());
  TEST_ASSERT(writable_cancel_yield.ok());
  TEST_ASSERT(writable_cancel_ok);
  TEST_ASSERT(!writable_ready_result.ok());
  TEST_ASSERT(!writable_ready_result.ready());
  TEST_ASSERT(!writable_ready_result.timed_out());
  TEST_ASSERT(writable_ready_result.code() == rund::ReasonCode::TaskCancelled);
  TEST_ASSERT(writable_close_result.ok());
  TEST_ASSERT(writable_post_close_scope.ok());
  TEST_ASSERT(writable_post_close_sleep.ok());
  TEST_ASSERT(writable_report.tasks().reactor().waits_canceled() >= 1u);
  TEST_ASSERT(writable_report.tasks().reactor().timeout_timer_cancels() >= 1u);
  TEST_ASSERT(writable_report.tasks().failed() == 0u);
  return 0;
}
