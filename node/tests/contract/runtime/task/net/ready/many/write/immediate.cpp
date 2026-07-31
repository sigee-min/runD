#include "src/host/net/test/ticket.hpp"
#include "src/host/net/test/socket.hpp"
#include "test/assert.hpp"

#include "../local.hpp"

#include <rund/net/bytes.hpp>
#include <rund/net/ready/many.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/socket.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include <array>
#include <span>

int RunNetReadyManyWriteImmediateOnlyCase() {
  SocketPairCleanup cleanup{};
  TEST_ASSERT(MakeSocketPair(cleanup));
  rund::net::Socket reader = rund::node::test::net::admit(cleanup.left);
  rund::net::Socket writer = rund::node::test::net::admit(cleanup.right);
  TEST_ASSERT(rund::net::nonblocking(reader.view(), true).ok());
  TEST_ASSERT(rund::net::nonblocking(writer.view(), true).ok());
  const std::array<rund::net::ready::Request, 1u> requests{
      rund::net::ready::Request{.socket = reader.view(),
                                .interest =
                                    rund::net::ready::Interest::Readable}};
  std::array<rund::net::ready::Event, 1u> events{};
  std::array<std::byte, 1u> payload{std::byte{'i'}};
  rund::net::ready::many::Result result{};
  rund::net::SendResult send{};
  rund::task::Status yielded{};
  rund::task::Status joined{};
  const rund::Session::Result report = rund::run(ReadyManyRunSpec(), [&] {
    auto wait = [&]() -> rund::task::Task<void> {
      result = co_await rund::net::ready::many::wait(
          std::span<const rund::net::ready::Request>{requests},
          std::span<rund::net::ready::Event>{events});
    };
    const rund::task::Handle waiter =
        rund::task::spawn("net-ready-many-immediate-only", wait());
    auto send_after_yield = [&]() -> rund::task::Task<void> {
      yielded = co_await rund::task::yield();
      send = rund::net::send(
          rund::node::test::net::ticket(writer.view(),
                                        rund::net::ready::Interest::Writable),
          std::span<const std::byte>{payload});
    };
    const rund::task::Handle send_task = rund::task::spawn(
        "net-ready-many-immediate-only-writer", send_after_yield());
    joined = rund::task::join(waiter, send_task);
  });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(yielded.ok());
  TEST_ASSERT(send.ok());
  TEST_ASSERT(result.ok());
  TEST_ASSERT(result.events == 0u);
  TEST_ASSERT(!result.timed_out());
  TEST_ASSERT(report.tasks().reactor_waits() == 0u);
  TEST_ASSERT(report.tasks().timers() == 0u);
  return 0;
}
