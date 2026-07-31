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
#include <utility>

int RunNetReadyManyWriteReadWriteCase() {
  SocketPairCleanup cleanup{};
  TEST_ASSERT(MakeSocketPair(cleanup));
  rund::net::Socket reader = rund::node::test::net::admit(cleanup.left);
  rund::net::Socket writer = rund::node::test::net::admit(cleanup.right);
  TEST_ASSERT(rund::net::nonblocking(reader.view(), true).ok());
  TEST_ASSERT(rund::net::nonblocking(writer.view(), true).ok());
  std::array<std::byte, 1u> payload{std::byte{'r'}};
  TEST_ASSERT(rund::net::direct::send(writer.view(),
                                      std::span<const std::byte>{payload})
                  .ok());
  const std::array<rund::net::ready::Request, 1u> requests{
      rund::net::ready::Request{.socket = reader.view(),
                                .interest =
                                    rund::net::ready::Interest::ReadWrite}};
  std::array<rund::net::ready::Event, 1u> events{};
  rund::net::ready::many::Result result{};
  std::array<std::byte, 1u> received{};
  rund::net::ReceiveResult receive{};
  rund::task::Status joined{};
  const rund::Session::Result report = rund::run(ReadyManyRunSpec(), [&] {
    auto wait = [&]() -> rund::task::Task<void> {
      result = co_await rund::net::ready::many::wait(
          std::span<const rund::net::ready::Request>{requests},
          std::span<rund::net::ready::Event>{events});
      if (result.ok() && result.events == 1u) {
        receive = rund::net::receive(std::move(events[0u].ticket), received);
      }
    };
    const rund::task::Handle waiter =
        rund::task::spawn("net-ready-many-write-both", wait());
    joined = rund::task::join(waiter);
  });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(result.ok());
  TEST_ASSERT(result.events == 1u);
  TEST_ASSERT(events[0u].index == 0u);
  TEST_ASSERT(events[0u].ticket.id() == reader.id());
  TEST_ASSERT(events[0u].ticket.interest() ==
              rund::net::ready::Interest::ReadWrite);
  TEST_ASSERT(events[0u].ticket.consumed());
  TEST_ASSERT(receive);
  TEST_ASSERT(receive.bytes == 1);
  TEST_ASSERT(received == payload);
  return 0;
}
