#include "src/host/net/test/socket.hpp"
#include "local.hpp"

#include <rund/net/ready/ticket.hpp>
#include <rund/net/ready/timed.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>

#include "test/assert.hpp"

#include <chrono>
#include <utility>

#include <sys/socket.h>

int RunTimedReadinessLeafCase() {
  TimedSocketPairCleanup ready_cleanup{};
  TimedSocketPairCleanup pending_cleanup{};
  TEST_ASSERT(MakeTimedSocketPair(ready_cleanup));
  TEST_ASSERT(MakeTimedSocketPair(pending_cleanup));

  const char byte = 'l';
  TEST_ASSERT(::send(ready_cleanup.right, &byte, 1u, 0) == 1);
  rund::net::Socket ready_reader =
      rund::node::test::net::admit(ready_cleanup.left);
  rund::net::Socket pending_reader =
      rund::node::test::net::admit(pending_cleanup.left);
  TEST_ASSERT(ready_reader);
  TEST_ASSERT(pending_reader);
  TEST_ASSERT(rund::net::nonblocking(ready_reader.view(), true).ok());
  TEST_ASSERT(rund::net::nonblocking(pending_reader.view(), true).ok());

  rund::net::ready::Ticket ready_result{};
  rund::net::ready::Ticket pending_result{};
  rund::task::Status ready_joined{};
  rund::task::Status pending_joined{};
  const rund::Session::Result report =
      rund::run(NetTimedReadinessRunSpec(), [&] {
        const rund::task::Handle ready_leaf = rund::task::spawn(
            "timed-ready-leaf-forbidden", [&] {
              ready_result = std::move(rund::net::ready::timed::read(
                                           ready_reader.view(),
                                           std::chrono::seconds{1}))
                                 .wait();
            });
        ready_joined = rund::task::join(ready_leaf);

        const rund::task::Handle pending_leaf = rund::task::spawn(
            "timed-pending-leaf-forbidden", [&] {
              pending_result = std::move(rund::net::ready::timed::read(
                                             pending_reader.view(),
                                             std::chrono::seconds{1}))
                                   .wait();
            });
        pending_joined = rund::task::join(pending_leaf);
      });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(!ready_result.ok());
  TEST_ASSERT(ready_result.code() ==
              rund::ReasonCode::TaskLeafPrimitiveForbidden);
  TEST_ASSERT(!pending_result.ok());
  TEST_ASSERT(pending_result.code() ==
              rund::ReasonCode::TaskLeafPrimitiveForbidden);
  TEST_ASSERT(!ready_joined.ok());
  TEST_ASSERT(ready_joined.code() ==
              rund::ReasonCode::TaskLeafPrimitiveForbidden);
  TEST_ASSERT(!pending_joined.ok());
  TEST_ASSERT(pending_joined.code() ==
              rund::ReasonCode::TaskLeafPrimitiveForbidden);
  TEST_ASSERT(report.tasks().failed() == 2u);
  TEST_ASSERT(report.tasks().completed() == 0u);
  TEST_ASSERT(report.tasks().reactor_waits() == 0u);
  TEST_ASSERT(report.tasks().timers() == 0u);
  TEST_ASSERT(report.tasks().observations() == 0u);
  TEST_ASSERT(report.tasks().host_events() == 0u);
  return 0;
}
