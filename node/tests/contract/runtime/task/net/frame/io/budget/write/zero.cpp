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

bool FrameIoZeroWriteBudgetReportsIncomplete() {
  const std::array<std::byte, 3u> payload{std::byte{'b'}, std::byte{'u'},
                                          std::byte{'d'}};
  FrameIoSocketPair pair{};
  FRAMEIO_ASSERT(OpenFrameIoNonblockingPair(pair));
  rund::net::frame::WriteResult written{};
  rund::task::Status joined{};
  const rund::Session::Result report = rund::run([&] {
    auto write = [&]() -> rund::task::Task<void> {
      const auto result = co_await rund::net::frame::write(
          pair.left.view(), std::span<const std::byte>{payload},
          rund::net::frame::IoLimit{.max_writes = 0u});
      if (result) {
        written = *result;
      }
    };
    const rund::task::Handle task =
        rund::task::spawn("net-frame-io-zero-write-budget", write());
    joined = rund::task::join(task);
  });

  FRAMEIO_ASSERT(report.ok());
  FRAMEIO_ASSERT(joined.ok());
  FRAMEIO_ASSERT(written.ok());
  FRAMEIO_ASSERT(!written.complete());
  FRAMEIO_ASSERT(written.budget_exhausted);
  FRAMEIO_ASSERT(!written.header_written);
  FRAMEIO_ASSERT(!written.payload_written);
  FRAMEIO_ASSERT(written.header_bytes == 0u);
  FRAMEIO_ASSERT(written.payload_bytes == 0u);
  FRAMEIO_ASSERT(written.bytes == 0u);

  auto ticket = rund::node::test::net::ticket(
      pair.left.view(), rund::net::ready::Interest::Writable);
  const rund::net::frame::WriteResult direct = rund::net::frame::write(
      std::move(ticket), std::span<const std::byte>{payload},
      rund::net::frame::IoLimit{.max_writes = 0u});
  FRAMEIO_ASSERT(direct);
  FRAMEIO_ASSERT(direct.budget_exhausted);
  FRAMEIO_ASSERT(ticket.consumed());
  return true;
}
