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
#include <chrono>
#include <cstdint>
#include <span>

int RunNetReadyManyConnectMultiInterestCase() {
  SocketPairCleanup cleanup{};
  TEST_ASSERT(MakeSocketPair(cleanup));
  rund::net::Socket reader = rund::node::test::net::admit(cleanup.left);
  TEST_ASSERT(rund::net::nonblocking(reader.view(), true).ok());
  std::array<std::byte, 4096u> fill{};
  bool would_block = false;
  for (std::uint32_t fill_attempt = 0u; fill_attempt < 4096u; ++fill_attempt) {
    const rund::net::SendResult fill_result = rund::net::send(
        rund::node::test::net::ticket(reader.view(),
                                      rund::net::ready::Interest::Writable),
        std::span<const std::byte>{fill});
    if (!fill_result.ok() &&
        fill_result.code() == rund::ReasonCode::IoWouldBlock) {
      would_block = true;
      break;
    }
    TEST_ASSERT(fill_result.ok());
  }
  TEST_ASSERT(would_block);

  const std::array<rund::net::ready::Request, 2u> requests{
      rund::net::ready::Request{.socket = reader.view(),
                                .interest =
                                    rund::net::ready::Interest::Readable},
      rund::net::ready::Request{.socket = reader.view(),
                                .interest =
                                    rund::net::ready::Interest::Writable},
  };
  std::array<rund::net::ready::Event, 2u> events{};
  rund::net::ready::many::Result result{};
  rund::net::CloseResult close_result{};
  rund::task::Status yielded{};
  rund::task::Status joined{};
  const rund::Session::Result report =
      rund::run(ReadyManyRunSpec(4u, 8u, 16u), [&] {
        auto wait = [&]() -> rund::task::Task<void> {
          result = co_await rund::net::ready::many::wait(
              std::span<const rund::net::ready::Request>{requests},
              std::span<rund::net::ready::Event>{events},
              std::chrono::seconds{30});
        };
        const rund::task::Handle waiter =
            rund::task::spawn("net-ready-many-multi-interest-invalid", wait());
        auto close = [&]() -> rund::task::Task<void> {
          yielded = co_await rund::task::yield();
          close_result = reader.close();
        };
        const rund::task::Handle closer = rund::task::spawn(
            "net-ready-many-multi-interest-invalid-closer", close());
        joined = rund::task::join(waiter, closer);
      });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(yielded.ok());
  TEST_ASSERT(close_result.ok());
  TEST_ASSERT(!result.ok());
  TEST_ASSERT(result.code() == rund::ReasonCode::IoFdInvalid);
  TEST_ASSERT(result.events == 1u);
  TEST_ASSERT(events[0u].ticket.code() == rund::ReasonCode::IoFdInvalid);
  TEST_ASSERT(events[0u].ticket.id() == requests[0u].socket.id());
  return 0;
}
