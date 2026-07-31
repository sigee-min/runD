#include "src/host/net/test/ticket.hpp"
#include "../../local.hpp"
#include <rund/net/frame/io.hpp>
#include <rund/net/frame/limit.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/task/api.hpp>

#include <array>
#include <cstddef>
#include <span>
#include <utility>

bool FrameIoZeroReadBudgetReportsIncomplete() {
  FrameIoSocketPair pair{};
  FRAMEIO_ASSERT(OpenFrameIoNonblockingPair(pair));
  std::array<std::byte, 3u> received{};
  rund::net::frame::ReadResult read{};
  rund::task::Status joined{};
  const rund::Session::Result report = rund::run([&] {
    auto receive = [&]() -> rund::task::Task<void> {
      const auto result = co_await rund::net::frame::read(
          pair.right.view(), std::span<std::byte>{received},
          rund::net::frame::IoLimit{.max_reads = 0u});
      if (result) {
        read = *result;
      }
    };
    const rund::task::Handle task =
        rund::task::spawn("net-frame-io-zero-read-budget", receive());
    joined = rund::task::join(task);
  });

  FRAMEIO_ASSERT(report.ok());
  FRAMEIO_ASSERT(joined.ok());
  FRAMEIO_ASSERT(read.ok());
  FRAMEIO_ASSERT(!read.complete());
  FRAMEIO_ASSERT(read.budget_exhausted);
  FRAMEIO_ASSERT(!read.header_read);
  FRAMEIO_ASSERT(!read.payload_read);
  FRAMEIO_ASSERT(read.header_bytes == 0u);
  FRAMEIO_ASSERT(read.payload_bytes == 0u);
  FRAMEIO_ASSERT(read.bytes == 0u);

  auto ticket = rund::node::test::net::ticket(
      pair.right.view(), rund::net::ready::Interest::Readable);
  const rund::net::frame::ReadResult direct =
      rund::net::frame::read(std::move(ticket), std::span<std::byte>{received},
                             rund::net::frame::IoLimit{.max_reads = 0u});
  FRAMEIO_ASSERT(direct);
  FRAMEIO_ASSERT(direct.budget_exhausted);
  FRAMEIO_ASSERT(ticket.consumed());
  return true;
}
