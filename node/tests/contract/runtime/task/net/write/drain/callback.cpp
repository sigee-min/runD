#include "local.hpp"
#include "src/host/net/test/socket.hpp"
#include <rund/task/api.hpp>

#include "test/assert.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

int RunWriteDrainCallbackCase() {
  constexpr std::size_t kPayloadSize = 1024u;
  const auto payload = MakeWriteDrainPayload<kPayloadSize>();

  WriteDrainSocketPairCleanup callback_cleanup{};
  TEST_ASSERT(MakeWriteDrainSocketPair(callback_cleanup));
  rund::net::Socket callback_writer =
      rund::node::test::net::admit(callback_cleanup.right);
  TEST_ASSERT(rund::net::nonblocking(callback_writer.view(), true).ok());
  rund::net::drain::WriteResult callback_drained{};
  rund::net::SendResult callback_send{};
  rund::net::CloseResult callback_close{};
  std::uint32_t callback_readers = ~std::uint32_t{0u};
  std::uint64_t callback_offset = 0u;
  std::uint64_t callback_calls = 0u;
  rund::task::Status callback_joined{};
  const rund::Session::Result callback_report =
      rund::run(NetWriteDrainRunSpec(), [&] {
        auto write = [&]() -> rund::task::Task<void> {
          rund::net::ready::Ticket writable =
              co_await rund::net::ready::write(callback_writer.view());
          callback_drained = rund::net::drain::write(
              std::move(writable),
              std::span<const std::byte>{payload}.first(4u),
              rund::net::drain::Budget{.max_operations = 4u},
              [&](const std::uint64_t completed_offset,
                  const rund::net::SendResult result) {
                ++callback_calls;
                callback_offset = completed_offset;
                callback_send = result;
                callback_readers =
                    rund::node::test::net::reader_count(callback_writer.view());
                if (callback_readers == 0u) {
                  callback_close = callback_writer.close();
                }
                return false;
              });
          co_return;
        };
        const rund::task::Handle callback_writer_task =
            rund::task::spawn("net-write-drain-callback", write());
        callback_joined = rund::task::join(callback_writer_task);
      });

  TEST_ASSERT(callback_report.ok());
  TEST_ASSERT(callback_joined.ok());
  TEST_ASSERT(callback_drained.ok());
  TEST_ASSERT(callback_drained.handler_stopped);
  TEST_ASSERT(callback_calls == 1u);
  TEST_ASSERT(callback_send.ok());
  TEST_ASSERT(callback_offset == callback_drained.bytes);
  TEST_ASSERT(callback_drained.bytes > 0u);
  TEST_ASSERT(callback_close.ok());
  TEST_ASSERT(!callback_writer);
  TEST_ASSERT(callback_readers == 0u);
  TEST_ASSERT(callback_report.tasks().network().send_calls() == 1u);
  TEST_ASSERT(callback_report.events().size() >= 2u);
  TEST_ASSERT(
      callback_report.events()[callback_report.events().size() - 2u].kind ==
      rund::host::EventKind::NetSend);
  TEST_ASSERT(callback_report.events().back().kind ==
              rund::host::EventKind::IoClose);
  return 0;
}
