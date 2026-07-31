#include "src/host/net/test/socket.hpp"
#include "local.hpp"
#include <rund/net/ready/ticket.hpp>
#include <rund/net/ready/timed.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>

#include "test/assert.hpp"

#include <chrono>

int RunTimedReadinessInvalidCase() {
  TimedSocketPairCleanup invalid_cleanup{};
  TEST_ASSERT(MakeTimedSocketPair(invalid_cleanup));
  rund::net::Socket invalid_reader =
      rund::node::test::net::admit(invalid_cleanup.left);
  TEST_ASSERT(rund::net::nonblocking(invalid_reader.view(), true).ok());
  rund::net::ready::Ticket invalid_result{};
  rund::task::Status invalid_joined{};
  const rund::Session::Result invalid_report =
      rund::run(NetTimedReadinessRunSpec(), [&] {
        auto wait = [&]() -> rund::task::Task<void> {
          invalid_result = co_await rund::net::ready::timed::read(
              invalid_reader.view(), std::chrono::nanoseconds{-1});
        };
        const rund::task::Handle waiter =
            rund::task::spawn("net-readable-invalid-timeout", wait());
        invalid_joined = rund::task::join(waiter);
      });

  TEST_ASSERT(invalid_report.ok());
  TEST_ASSERT(invalid_joined.ok());
  TEST_ASSERT(!invalid_result.ok());
  TEST_ASSERT(invalid_result.code() == rund::ReasonCode::TimerDurationInvalid);
  return 0;
}
