#include "src/host/net/test/socket.hpp"
#include "local.hpp"
#include <rund/task/api.hpp>

#include "test/assert.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

int RunWriteDrainSuccessCase() {
  WriteDrainSocketPairCleanup cleanup{};
  TEST_ASSERT(MakeWriteDrainSocketPair(cleanup));
  rund::net::Socket reader = rund::node::test::net::admit(cleanup.left);
  rund::net::Socket writer = rund::node::test::net::admit(cleanup.right);
  TEST_ASSERT(rund::net::nonblocking(reader.view(), true).ok());
  TEST_ASSERT(rund::net::nonblocking(writer.view(), true).ok());

  constexpr std::size_t kPayloadSize = 1024u;
  const std::array<std::byte, kPayloadSize> payload =
      MakeWriteDrainPayload<kPayloadSize>();
  std::array<std::byte, kPayloadSize> received{};
  std::uint64_t received_bytes = 0u;
  rund::net::drain::WriteResult drained{};
  rund::net::ready::Ticket last_readiness{};
  rund::net::ReceiveResult last_receive{};
  rund::task::Status joined{};

  const rund::Session::Result report = rund::run(NetWriteDrainRunSpec(), [&] {
    auto write = [&]() -> rund::task::Task<void> {
      auto writable = co_await rund::net::ready::write(writer.view());
      drained = rund::net::drain::write(
          std::move(writable), std::span<const std::byte>{payload},
          rund::net::drain::Budget{.max_operations = 64u});
      co_return;
    };
    const rund::task::Handle writer_task =
        rund::task::spawn("net-write-drain-writer", write());
    auto read = [&]() -> rund::task::Task<void> {
      while (received_bytes < payload.size()) {
        last_readiness = co_await rund::net::ready::read(reader.view());
        if (!last_readiness.ok()) {
          co_return;
        }
        last_receive = rund::net::receive(
            std::move(last_readiness),
            std::span<std::byte>{received}.subspan(received_bytes));
        if (!last_receive.ok()) {
          if (last_receive.code() == rund::ReasonCode::IoWouldBlock) {
            continue;
          }
          co_return;
        }
        received_bytes += static_cast<std::uint64_t>(last_receive.bytes);
      }
    };
    const rund::task::Handle reader_task =
        rund::task::spawn("net-write-drain-reader", read());
    joined = rund::task::join(writer_task, reader_task);
  });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(drained.ok());
  TEST_ASSERT(drained.all_written);
  TEST_ASSERT(drained.bytes == payload.size());
  TEST_ASSERT(drained.writes > 0u);
  TEST_ASSERT(received_bytes == payload.size());
  TEST_ASSERT(received == payload);
  return 0;
}
