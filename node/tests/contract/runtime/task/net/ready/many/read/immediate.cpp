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
#include <cstddef>
#include <span>

int RunNetReadyManyReadImmediateCase() {
  std::array<SocketPairCleanup, 3u> cleanup{};
  std::array<rund::net::Socket, 3u> readers{};
  std::array<rund::net::Socket, 3u> writers{};
  for (std::size_t index = 0u; index < cleanup.size(); ++index) {
    TEST_ASSERT(MakeSocketPair(cleanup[index]));
    readers[index] = rund::node::test::net::admit(cleanup[index].left);
    writers[index] = rund::node::test::net::admit(cleanup[index].right);
    TEST_ASSERT(rund::net::nonblocking(readers[index].view(), true).ok());
    TEST_ASSERT(rund::net::nonblocking(writers[index].view(), true).ok());
  }

  std::array<std::byte, 1u> payload{std::byte{'m'}};
  TEST_ASSERT(rund::net::send(
                  rund::node::test::net::ticket(
                      writers[2u].view(), rund::net::ready::Interest::Writable),
                  std::span<const std::byte>{payload})
                  .ok());
  TEST_ASSERT(rund::net::send(
                  rund::node::test::net::ticket(
                      writers[0u].view(), rund::net::ready::Interest::Writable),
                  std::span<const std::byte>{payload})
                  .ok());

  std::array<rund::net::ready::Request, 3u> requests{
      rund::net::ready::Request{.socket = readers[0u].view(),
                                .interest =
                                    rund::net::ready::Interest::Readable},
      rund::net::ready::Request{.socket = readers[1u].view(),
                                .interest =
                                    rund::net::ready::Interest::Readable},
      rund::net::ready::Request{.socket = readers[2u].view(),
                                .interest =
                                    rund::net::ready::Interest::Readable},
  };
  std::array<rund::net::ready::Event, 3u> events{};
  rund::net::ready::many::Result result{};
  rund::task::Status joined{};
  const rund::Session::Result report = rund::run(ReadyManyRunSpec(), [&] {
    auto wait = [&]() -> rund::task::Task<void> {
      result = co_await rund::net::ready::many::wait(
          std::span<const rund::net::ready::Request>{requests},
          std::span<rund::net::ready::Event>{events});
    };
    const rund::task::Handle waiter =
        rund::task::spawn("net-ready-many-immediate", wait());
    joined = rund::task::join(waiter);
  });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(result.ok());
  TEST_ASSERT(result.events == 2u);
  TEST_ASSERT(!result.timed_out());
  TEST_ASSERT(!result.budget_exhausted);
  TEST_ASSERT(events[0u].index == 0u);
  TEST_ASSERT(events[1u].index == 2u);
  TEST_ASSERT(events[0u].ticket.id() == readers[0u].id());
  TEST_ASSERT(events[1u].ticket.id() == readers[2u].id());
  return 0;
}
