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
#include <string_view>

int RunNetCancellationReadableWakeCase() {
  CancelSocketPairCleanup wake_cleanup{};
  TEST_ASSERT(MakeCancelSocketPair(wake_cleanup));
  rund::net::Socket wake_reader =
      rund::node::test::net::admit(wake_cleanup.left);
  rund::net::Socket wake_writer =
      rund::node::test::net::admit(wake_cleanup.right);
  TEST_ASSERT(rund::net::nonblocking(wake_reader.view(), true).ok());
  TEST_ASSERT(rund::net::nonblocking(wake_writer.view(), true).ok());

  bool wake_source_valid = false;
  bool wake_token_valid = false;
  rund::net::ready::Ticket wake_ready_result{};
  bool wake_cancel_ok = false;
  rund::task::Status wake_cancel_yield{};
  rund::task::Status wake_join{};
  const rund::Session::Result wake_report =
      rund::run(NetCancellationRunSpec(), [&] {
        auto source = rund::task::stop_source::create();
        wake_source_valid = static_cast<bool>(source);
        if (!wake_source_valid) {
          return;
        }
        auto token = source.token();
        rund::task::stop_token explicit_token = token;
        wake_token_valid = static_cast<bool>(explicit_token);
        if (!wake_token_valid) {
          return;
        }

        auto wait = [&]() -> rund::task::Task<void> {
          wake_ready_result = co_await rund::net::ready::timed::read(
              wake_reader.view(), std::chrono::seconds{30}, explicit_token);
        };
        const rund::task::Handle waiter =
            rund::task::spawn("net-cancel-readable-waiter", wait());
        auto cancel = [&]() -> rund::task::Task<void> {
          wake_cancel_yield = co_await rund::task::yield();
          wake_cancel_ok = source.request_stop().ok();
        };
        const rund::task::Handle canceller =
            rund::task::spawn("net-cancel-readable-requester", cancel());
        wake_join = rund::task::join(waiter, canceller);
      });

  TEST_ASSERT(wake_report.ok());
  TEST_ASSERT(wake_source_valid);
  TEST_ASSERT(wake_token_valid);
  TEST_ASSERT(wake_join.ok());
  TEST_ASSERT(wake_cancel_yield.ok());
  TEST_ASSERT(wake_cancel_ok);
  TEST_ASSERT(!wake_ready_result.ok());
  TEST_ASSERT(!wake_ready_result.ready());
  TEST_ASSERT(!wake_ready_result.timed_out());
  TEST_ASSERT(wake_ready_result.code() == rund::ReasonCode::TaskCancelled);
  TEST_ASSERT(std::string_view{wake_ready_result.error()} == "task_cancelled");
  return 0;
}
