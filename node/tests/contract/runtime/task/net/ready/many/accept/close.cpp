#include "src/host/net/test/ticket.hpp"
#include "src/host/net/test/socket.hpp"
#include "local.hpp"
#include <rund/net/bytes.hpp>
#include <rund/net/ready/many.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include "test/assert.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <span>

int RunReadyManyAcceptCloseCase() {
  constexpr std::size_t close_count = 4u;
  std::array<SocketPairCleanup, close_count> close_cleanup{};
  std::array<rund::net::Socket, close_count> close_readers{};
  std::array<rund::net::Socket, close_count> close_writers{};
  std::array<rund::net::ready::Request, close_count> close_requests{};
  std::array<rund::net::ready::Event, close_count> close_events{};
  SocketPairCleanup close_scratch_cleanup{};
  TEST_ASSERT(MakeSocketPair(close_scratch_cleanup));
  rund::net::Socket close_scratch_reader =
      rund::node::test::net::admit(close_scratch_cleanup.left);
  rund::net::Socket close_scratch_writer =
      rund::node::test::net::admit(close_scratch_cleanup.right);
  TEST_ASSERT(rund::net::nonblocking(close_scratch_reader.view(), true).ok());
  TEST_ASSERT(rund::net::nonblocking(close_scratch_writer.view(), true).ok());
  std::array<std::byte, 1u> close_scratch_byte{std::byte{'s'}};
  TEST_ASSERT(rund::net::send(rund::node::test::net::ticket(
                                  close_scratch_writer.view(),
                                  rund::net::ready::Interest::Writable),
                              std::span<const std::byte>{close_scratch_byte})
                  .ok());
  const std::array<rund::net::ready::Request, 1u> close_scratch_requests{
      rund::net::ready::Request{.socket = close_scratch_reader.view(),
                                .interest =
                                    rund::net::ready::Interest::Readable}};
  std::array<rund::net::ready::Event, 1u> close_scratch_events{};
  for (std::size_t index = 0u; index < close_count; ++index) {
    TEST_ASSERT(MakeSocketPair(close_cleanup[index]));
    close_readers[index] =
        rund::node::test::net::admit(close_cleanup[index].left);
    close_writers[index] =
        rund::node::test::net::admit(close_cleanup[index].right);
    TEST_ASSERT(rund::net::nonblocking(close_readers[index].view(), true).ok());
    TEST_ASSERT(rund::net::nonblocking(close_writers[index].view(), true).ok());
    close_requests[index] = rund::net::ready::Request{
        .socket = close_readers[index].view(),
        .interest = rund::net::ready::Interest::Readable};
  }

  rund::net::ready::many::Result close_result{};
  rund::net::ready::many::Result close_scratch_result{};
  rund::net::CloseResult close_status{};
  rund::task::Status close_yielded{};
  rund::task::Status close_joined{};
  const rund::Session::Result close_report =
      rund::run(ReadyManyRunSpec(4u, 8u, 16u), [&] {
        auto wait = [&]() -> rund::task::Task<void> {
          close_result = co_await rund::net::ready::many::wait(
              std::span<const rund::net::ready::Request>{close_requests},
              std::span<rund::net::ready::Event>{close_events},
              std::chrono::seconds{30});
        };
        const rund::task::Handle waiter =
            rund::task::spawn("net-ready-many-close-invalidation", wait());
        auto close = [&]() -> rund::task::Task<void> {
          close_yielded = co_await rund::task::yield();
          close_status = close_readers[2u].close();
          close_scratch_result = co_await rund::net::ready::many::wait(
              std::span<const rund::net::ready::Request>{
                  close_scratch_requests},
              std::span<rund::net::ready::Event>{close_scratch_events});
        };
        const rund::task::Handle closer = rund::task::spawn(
            "net-ready-many-close-invalidation-closer", close());
        close_joined = rund::task::join(waiter, closer);
      });

  TEST_ASSERT(close_report.ok());
  TEST_ASSERT(close_joined.ok());
  TEST_ASSERT(close_yielded.ok());
  TEST_ASSERT(close_status.ok());
  TEST_ASSERT(close_scratch_result.ok());
  TEST_ASSERT(close_scratch_result.events == 1u);
  TEST_ASSERT(!close_result.ok());
  TEST_ASSERT(close_result.code() == rund::ReasonCode::IoFdInvalid);
  TEST_ASSERT(close_result.events == 1u);
  TEST_ASSERT(close_events[0u].index == 2u);
  TEST_ASSERT(close_events[0u].ticket.code() == rund::ReasonCode::IoFdInvalid);
  return 0;
}
