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
#include <utility>

int RunReadyManyAcceptBlockingCase() {
  constexpr std::size_t blocking_count = 64u;
  std::array<SocketPairCleanup, blocking_count> blocking_cleanup{};
  std::array<rund::net::Socket, blocking_count> blocking_readers{};
  std::array<rund::net::Socket, blocking_count> blocking_writers{};
  std::array<rund::net::ready::Request, blocking_count> blocking_requests{};
  std::array<rund::net::ready::Event, blocking_count> blocking_events{};
  for (std::size_t index = 0u; index < blocking_count; ++index) {
    TEST_ASSERT(MakeSocketPair(blocking_cleanup[index]));
    blocking_readers[index] =
        rund::node::test::net::admit(blocking_cleanup[index].left);
    blocking_writers[index] =
        rund::node::test::net::admit(blocking_cleanup[index].right);
    TEST_ASSERT(
        rund::net::nonblocking(blocking_readers[index].view(), true).ok());
    TEST_ASSERT(
        rund::net::nonblocking(blocking_writers[index].view(), true).ok());
    blocking_requests[index] = rund::net::ready::Request{
        .socket = blocking_readers[index].view(),
        .interest = rund::net::ready::Interest::Readable};
  }

  std::array<std::byte, 1u> blocking_byte{std::byte{'b'}};
  rund::net::ready::many::Result blocking_result{};
  rund::net::ready::many::Result deferred_wait_result{};
  rund::net::ready::many::Result moved_from_result{};
  rund::net::SendResult blocking_send{};
  rund::task::Status blocking_yielded{};
  rund::task::Status blocking_joined{};
  bool blocking_writes_ok = true;
  const rund::Session::Result blocking_report =
      rund::run(ReadyManyRunSpec(4u, 8u, 96u), [&] {
        auto wait = [&]() -> rund::task::Task<void> {
          auto operation = rund::net::ready::many::wait(
              std::span<const rund::net::ready::Request>{blocking_requests},
              std::span<rund::net::ready::Event>{blocking_events},
              std::chrono::milliseconds{100});
          deferred_wait_result = operation.wait();
          auto owner = std::move(operation);
          moved_from_result = co_await std::move(operation);
          blocking_result = co_await std::move(owner);
        };
        const rund::task::Handle waiter =
            rund::task::spawn("net-ready-many-blocking", wait());
        auto write = [&]() -> rund::task::Task<void> {
          blocking_yielded = co_await rund::task::yield();
          for (std::size_t index = 0u; index < blocking_count; index += 4u) {
            blocking_send =
                rund::net::send(rund::node::test::net::ticket(
                                    blocking_writers[index].view(),
                                    rund::net::ready::Interest::Writable),
                                std::span<const std::byte>{blocking_byte});
            blocking_writes_ok = blocking_writes_ok && blocking_send.ok();
          }
        };
        const rund::task::Handle writer =
            rund::task::spawn("net-ready-many-blocking-writer", write());
        blocking_joined = rund::task::join(waiter, writer);
      });

  TEST_ASSERT(blocking_report.ok());
  TEST_ASSERT(blocking_joined.ok());
  TEST_ASSERT(blocking_yielded.ok());
  TEST_ASSERT(blocking_writes_ok);
  TEST_ASSERT(deferred_wait_result.code() ==
              rund::ReasonCode::TaskContextMissing);
  TEST_ASSERT(moved_from_result.code() == rund::ReasonCode::TaskInvalid);
  TEST_ASSERT(blocking_result.ok());
  TEST_ASSERT(blocking_result.events == 16u);
  TEST_ASSERT(!blocking_result.timed_out());
  TEST_ASSERT(blocking_report.tasks().reactor_waits() == blocking_count);
  for (std::uint32_t event_index = 0u; event_index < blocking_result.events;
       ++event_index) {
    TEST_ASSERT(blocking_events[event_index].index == event_index * 4u);
    TEST_ASSERT(blocking_events[event_index].ticket.id() ==
                blocking_readers[event_index * 4u].id());
  }
  return 0;
}
