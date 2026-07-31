#include "src/host/net/test/ticket.hpp"
#include "src/host/net/test/socket.hpp"
#include "test/assert.hpp"

#include "../local.hpp"

#include "src/runtime/task/scheduler/reactor/many/store.hpp"

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
#include <vector>

int RunNetReadyManyReadBatchCase() {
  std::array<SocketPairCleanup, 5u> cleanup{};
  std::array<rund::net::Socket, 5u> readers{};
  std::array<rund::net::Socket, 5u> writers{};
  for (std::size_t index = 0u; index < cleanup.size(); ++index) {
    TEST_ASSERT(MakeSocketPair(cleanup[index]));
    readers[index] = rund::node::test::net::admit(cleanup[index].left);
    writers[index] = rund::node::test::net::admit(cleanup[index].right);
    TEST_ASSERT(rund::net::nonblocking(readers[index].view(), true).ok());
    TEST_ASSERT(rund::net::nonblocking(writers[index].view(), true).ok());
  }

  std::array<std::byte, 1u> payload{std::byte{'q'}};
  TEST_ASSERT(rund::net::send(
                  rund::node::test::net::ticket(
                      writers[4u].view(), rund::net::ready::Interest::Writable),
                  std::span<const std::byte>{payload})
                  .ok());
  TEST_ASSERT(rund::net::send(
                  rund::node::test::net::ticket(
                      writers[1u].view(), rund::net::ready::Interest::Writable),
                  std::span<const std::byte>{payload})
                  .ok());
  TEST_ASSERT(rund::net::send(
                  rund::node::test::net::ticket(
                      writers[3u].view(), rund::net::ready::Interest::Writable),
                  std::span<const std::byte>{payload})
                  .ok());

  std::array<rund::net::ready::Request, 5u> requests{};
  for (std::size_t index = 0u; index < requests.size(); ++index) {
    requests[index] = rund::net::ready::Request{
        .socket = readers[index].view(),
        .interest = rund::net::ready::Interest::Readable};
  }
  std::array<rund::net::ready::Event, 5u> events{};
  rund::net::ready::many::Result result{};
  rund::task::Status joined{};
  const rund::Session::Result report = rund::run(ReadyManyRunSpec(), [&] {
    auto wait = [&]() -> rund::task::Task<void> {
      result = co_await rund::net::ready::many::wait(
          std::span<const rund::net::ready::Request>{requests},
          std::span<rund::net::ready::Event>{events});
    };
    const rund::task::Handle waiter =
        rund::task::spawn("net-ready-many-batch-immediate", wait());
    joined = rund::task::join(waiter);
  });
  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(result.ok());
  TEST_ASSERT(result.events == 3u);
  TEST_ASSERT(events[0u].index == 1u);
  TEST_ASSERT(events[1u].index == 3u);
  TEST_ASSERT(events[2u].index == 4u);
  return 0;
}

int RunNetReadyManyValidationScaleCase() {
  constexpr std::uint32_t kRequestCount = 1024u;
  std::vector<rund::node::ReactorManyRequest> requests(kRequestCount);
  for (std::uint32_t index = 0u; index < kRequestCount; ++index) {
    const std::uint32_t identity = (index * 641u) % kRequestCount;
    requests[index] = rund::node::ReactorManyRequest{
        .fd = static_cast<rund::node::ReactorHandle>(identity + 1u),
        .slot = index,
        .event_index = index,
        .interest = rund::node::ReactorInterest::Read,
    };
  }

  std::vector<std::uint32_t> order{};
  std::uint64_t comparisons = 0u;
  TEST_ASSERT(rund::node::ReactorManyValidateRequests(
                  requests, order, &comparisons) == rund::ReasonCode::Ok);
  const std::uint64_t quadratic_comparisons =
      static_cast<std::uint64_t>(kRequestCount) * (kRequestCount - 1u) / 2u;
  TEST_ASSERT(comparisons > 0u);
  TEST_ASSERT(comparisons * 8u < quadratic_comparisons);
  return 0;
}
