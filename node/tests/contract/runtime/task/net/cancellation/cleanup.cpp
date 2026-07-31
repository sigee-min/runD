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

int RunNetCancellationCleanupCase() {
  CancelSocketPairCleanup cleanup{};
  TEST_ASSERT(MakeCancelSocketPair(cleanup));
  rund::net::Socket cleanup_reader = rund::node::test::net::admit(cleanup.left);
  rund::net::Socket cleanup_writer =
      rund::node::test::net::admit(cleanup.right);
  TEST_ASSERT(rund::net::nonblocking(cleanup_reader.view(), true).ok());
  TEST_ASSERT(rund::net::nonblocking(cleanup_writer.view(), true).ok());

  bool cleanup_source_valid = false;
  bool cleanup_token_valid = false;
  rund::net::ready::Ticket cleanup_ready_result{};
  bool cleanup_cancel_ok = false;
  rund::task::Status cleanup_cancel_yield{};
  rund::net::CloseResult close_result{};
  rund::task::Status cleanup_scope{};
  rund::task::Status post_close_scope{};
  rund::task::Status post_close_sleep{};
  const rund::Session::Result cleanup_report =
      rund::run(NetCancellationRunSpec(), [&] {
        auto source = rund::task::stop_source::create();
        cleanup_source_valid = static_cast<bool>(source);
        if (!cleanup_source_valid) {
          return;
        }
        auto token = source.token();
        rund::task::stop_token explicit_token = token;
        cleanup_token_valid = static_cast<bool>(explicit_token);
        if (!cleanup_token_valid) {
          return;
        }

        auto wait = [&]() -> rund::task::Task<void> {
          cleanup_ready_result = co_await rund::net::ready::timed::read(
              cleanup_reader.view(), std::chrono::seconds{30}, explicit_token);
        };
        auto cancel = [&]() -> rund::task::Task<void> {
          cleanup_cancel_yield = co_await rund::task::yield();
          cleanup_cancel_ok = source.request_stop().ok();
        };
        cleanup_scope = rund::task::scope([&] {
          (void)rund::task::spawn("net-cancel-cleanup-waiter", wait());
          (void)rund::task::spawn("net-cancel-cleanup-requester", cancel());
        });

        close_result = cleanup_reader.close();

        auto sleep = [&]() -> rund::task::Task<void> {
          post_close_sleep =
              co_await rund::task::sleep(std::chrono::milliseconds{1});
        };
        post_close_scope = rund::task::scope([&] {
          (void)rund::task::spawn("net-cancel-cleanup-short-scope", sleep());
        });
      });

  TEST_ASSERT(cleanup_report.ok());
  TEST_ASSERT(cleanup_source_valid);
  TEST_ASSERT(cleanup_token_valid);
  TEST_ASSERT(cleanup_scope.ok());
  TEST_ASSERT(cleanup_cancel_yield.ok());
  TEST_ASSERT(cleanup_cancel_ok);
  TEST_ASSERT(!cleanup_ready_result.ok());
  TEST_ASSERT(cleanup_ready_result.code() == rund::ReasonCode::TaskCancelled);
  TEST_ASSERT(close_result.ok());
  TEST_ASSERT(post_close_scope.ok());
  TEST_ASSERT(post_close_sleep.ok());
  TEST_ASSERT(cleanup_report.tasks().reactor().waits_canceled() >= 1u);
  TEST_ASSERT(cleanup_report.tasks().reactor().timeout_timer_cancels() >= 1u);
  TEST_ASSERT(cleanup_report.tasks().failed() == 0u);
  return 0;
}
