#include "../../local.hpp"
#include <rund/net/frame/io.hpp>
#include <rund/net/frame/limit.hpp>
#include <rund/task/api.hpp>

#include <array>
#include <cstddef>
#include <span>

bool FrameIoHeaderOnlyReadBudgetReportsIncomplete() {
  const std::array<std::byte, 3u> payload{std::byte{'b'}, std::byte{'u'},
                                          std::byte{'d'}};
  FrameIoSocketPair pair{};
  FRAMEIO_ASSERT(OpenFrameIoNonblockingPair(pair));
  std::array<std::byte, 3u> received{};
  rund::net::frame::WriteResult written{};
  rund::net::frame::ReadResult read{};
  rund::task::Status joined{};
  const rund::Session::Result report = rund::run(FrameIoRunSpec(), [&] {
    auto exchange = [&]() -> rund::task::Task<void> {
      const auto write = co_await rund::net::frame::write(
          pair.left.view(), std::span<const std::byte>{payload});
      if (write) {
        written = *write;
      }
      const auto receive = co_await rund::net::frame::read(
          pair.right.view(), std::span<std::byte>{received},
          rund::net::frame::IoLimit{.max_reads = 1u});
      if (receive) {
        read = *receive;
      }
    };
    const rund::task::Handle task =
        rund::task::spawn("net-frame-io-one-read-budget", exchange());
    joined = rund::task::join(task);
  });

  FRAMEIO_ASSERT(report.ok());
  FRAMEIO_ASSERT(joined.ok());
  FRAMEIO_ASSERT(written);
  FRAMEIO_ASSERT(written.complete());
  FRAMEIO_ASSERT(read.ok());
  FRAMEIO_ASSERT(!read.complete());
  FRAMEIO_ASSERT(read.budget_exhausted);
  FRAMEIO_ASSERT(read.header_read);
  FRAMEIO_ASSERT(!read.payload_read);
  FRAMEIO_ASSERT(read.header_bytes == 4u);
  FRAMEIO_ASSERT(read.payload_bytes == 0u);
  FRAMEIO_ASSERT(read.bytes == 0u);
  return true;
}
