#include "src/host/net/test/ticket.hpp"
#include "src/host/net/test/socket.hpp"
#include "local.hpp"
#include <rund/task/api.hpp>

#include "test/assert.hpp"

#include <cstddef>
#include <span>
#include <utility>

int RunWriteDrainBoundCase() {
  constexpr std::size_t kPayloadSize = 1024u;
  const auto payload = MakeWriteDrainPayload<kPayloadSize>();

  WriteDrainSocketPairCleanup zero_cleanup{};
  TEST_ASSERT(MakeWriteDrainSocketPair(zero_cleanup));
  rund::net::Socket zero_writer =
      rund::node::test::net::admit(zero_cleanup.right);
  TEST_ASSERT(rund::net::nonblocking(zero_writer.view(), true).ok());
  rund::net::drain::WriteResult zero_drained{};
  bool zero_consumed = false;
  rund::task::Status zero_joined{};
  const rund::Session::Result zero_report =
      rund::run(NetWriteDrainRunSpec(), [&] {
        auto write = [&]() -> rund::task::Task<void> {
          auto ticket = rund::node::test::net::ticket(
              zero_writer.view(), rund::net::ready::Interest::Writable);
          zero_drained = rund::net::drain::write(
              std::move(ticket), std::span<const std::byte>{payload},
              rund::net::drain::Budget{.max_operations = 0u});
          zero_consumed = ticket.consumed();
          co_return;
        };
        const rund::task::Handle zero_writer_task =
            rund::task::spawn("net-write-drain-zero-budget", write());
        zero_joined = rund::task::join(zero_writer_task);
      });

  TEST_ASSERT(zero_report.ok());
  TEST_ASSERT(zero_joined.ok());
  TEST_ASSERT(zero_drained.ok());
  TEST_ASSERT(!zero_drained.all_written);
  TEST_ASSERT(zero_drained.budget_exhausted);
  TEST_ASSERT(zero_drained.bytes == 0u);
  TEST_ASSERT(zero_consumed);

  auto empty_ticket = rund::node::test::net::ticket(
      zero_writer.view(), rund::net::ready::Interest::Writable);
  const rund::net::drain::WriteResult empty_drained = rund::net::drain::write(
      std::move(empty_ticket), std::span<const std::byte>{},
      rund::net::drain::Budget{.max_operations = 0u});
  TEST_ASSERT(empty_drained.ok());
  TEST_ASSERT(empty_drained.all_written);
  TEST_ASSERT(!empty_drained.budget_exhausted);
  TEST_ASSERT(empty_drained.bytes == 0u);
  TEST_ASSERT(empty_ticket.consumed());
  return 0;
}
