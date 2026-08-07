#include "src/host/net/test/socket.hpp"
#include "src/host/net/test/ticket.hpp"
#include "test/assert.hpp"

#include <rund/net/bytes.hpp>
#include <rund/net/drain.hpp>
#include <rund/net/ready.hpp>
#include <rund/net/socket.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>

#include <array>
#include <cstddef>
#include <span>
#include <utility>

#include <sys/socket.h>

int RunRuntimeTaskNetDrainContract() {
  int sockets[2] = {-1, -1};
  TEST_ASSERT(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
  rund::net::Socket reader = rund::node::test::net::admit(sockets[0]);
  rund::net::Socket writer = rund::node::test::net::admit(sockets[1]);
  TEST_ASSERT(rund::net::nonblocking(reader.view(), true).ok());
  TEST_ASSERT(rund::net::nonblocking(writer.view(), true).ok());

  std::array<std::byte, 4u> buffer{};
  std::uint64_t callbacks = 0u;
  std::uint64_t bytes = 0u;
  rund::net::drain::ReadResult drained{};
  rund::task::Status joined{};

  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 4u,
                  .ready_queue_capacity = 4u,
                  .reactor_wait_capacity = 4u,
                  .observation_capacity = 32u,
                  .host_event_capacity = 32u,
              },
      },
      [&] {
        const std::array<std::byte, 3u> payload{
            std::byte{0x61}, std::byte{0x62}, std::byte{0x63}};
        const rund::net::SendResult sent = rund::net::send(
            rund::node::test::net::ticket(writer.view(),
                                          rund::net::ready::Interest::Writable),
            payload);
        if (!sent.ok() || sent.bytes != 3) {
          return;
        }
        auto read = [&]() -> rund::task::Task<void> {
          auto readable = co_await rund::net::ready::read(reader.view());
          drained = rund::net::drain::read(
              std::move(readable), buffer,
              rund::net::drain::Budget{.max_operations = 8u},
              [&](std::span<const std::byte> chunk) noexcept {
                ++callbacks;
                bytes += chunk.size();
                return true;
              });
          co_return;
        };
        const rund::task::Handle handle =
            rund::task::spawn("net-drain-reader", read());
        joined = rund::task::join(handle);
      });

  std::uint64_t zero_callbacks = 0u;
  auto zero_ticket = rund::node::test::net::ticket(
      reader.view(), rund::net::ready::Interest::Readable);
  const rund::net::drain::ReadResult zero =
      rund::net::drain::read(std::move(zero_ticket), buffer,
                             rund::net::drain::Budget{.max_operations = 0u},
                             [&](const std::span<const std::byte>) noexcept {
                               ++zero_callbacks;
                               return true;
                             });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(drained.ok());
  TEST_ASSERT(drained.would_block);
  TEST_ASSERT(callbacks == 1u);
  TEST_ASSERT(bytes == 3u);
  TEST_ASSERT(drained.reads == 1u);
  TEST_ASSERT(drained.bytes == 3u);
  TEST_ASSERT(zero);
  TEST_ASSERT(zero.budget_exhausted);
  TEST_ASSERT(zero.reads == 0u);
  TEST_ASSERT(zero.bytes == 0u);
  TEST_ASSERT(zero_callbacks == 0u);
  TEST_ASSERT(zero_ticket.consumed());

  int close_sockets[2] = {-1, -1};
  TEST_ASSERT(::socketpair(AF_UNIX, SOCK_STREAM, 0, close_sockets) == 0);
  rund::net::Socket close_reader =
      rund::node::test::net::admit(close_sockets[0]);
  rund::net::Socket close_writer =
      rund::node::test::net::admit(close_sockets[1]);
  TEST_ASSERT(rund::net::nonblocking(close_reader.view(), true).ok());
  TEST_ASSERT(rund::net::nonblocking(close_writer.view(), true).ok());
  rund::net::drain::ReadResult close_drained{};
  rund::net::CloseResult callback_close{};
  std::uint32_t callback_readers = ~std::uint32_t{0u};
  std::uint64_t close_callbacks = 0u;
  rund::task::Status close_joined{};
  const rund::Session::Result close_report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 4u,
                  .ready_queue_capacity = 4u,
                  .reactor_wait_capacity = 4u,
                  .observation_capacity = 32u,
                  .host_event_capacity = 32u,
              },
      },
      [&] {
        const std::array<std::byte, 1u> payload{std::byte{0x71}};
        const rund::net::SendResult sent = rund::net::send(
            rund::node::test::net::ticket(close_writer.view(),
                                          rund::net::ready::Interest::Writable),
            payload);
        if (!sent.ok() || sent.bytes != 1) {
          return;
        }
        auto read_then_close = [&]() -> rund::task::Task<void> {
          auto readable = co_await rund::net::ready::read(close_reader.view());
          close_drained = rund::net::drain::read(
              std::move(readable), buffer,
              rund::net::drain::Budget{.max_operations = 2u},
              [&](const std::span<const std::byte> chunk) noexcept {
                ++close_callbacks;
                callback_readers =
                    rund::node::test::net::reader_count(close_reader.view());
                if (callback_readers != 0u || chunk.size() != 1u) {
                  return false;
                }
                callback_close = close_reader.close();
                return true;
              });
          co_return;
        };
        const rund::task::Handle handle =
            rund::task::spawn("net-drain-close-reader", read_then_close());
        close_joined = rund::task::join(handle);
      });

  TEST_ASSERT(close_report.ok());
  TEST_ASSERT(close_joined.ok());
  TEST_ASSERT(callback_close.ok());
  TEST_ASSERT(!close_reader);
  TEST_ASSERT(callback_readers == 0u);
  TEST_ASSERT(close_callbacks == 1u);
  TEST_ASSERT(!close_drained.ok());
  TEST_ASSERT(close_drained.code() == rund::ReasonCode::IoFdInvalid);
  TEST_ASSERT(close_drained.reads == 1u);
  TEST_ASSERT(close_drained.bytes == 1u);
  TEST_ASSERT(!close_drained.would_block);
  TEST_ASSERT(!close_drained.budget_exhausted);
  TEST_ASSERT(!close_drained.handler_stopped);
  TEST_ASSERT(close_report.tasks().network().recv_calls() == 1u);
  TEST_ASSERT(close_report.events().size() >= 2u);
  TEST_ASSERT(close_report.events()[close_report.events().size() - 2u].kind ==
              rund::host::EventKind::NetRecv);
  TEST_ASSERT(close_report.events().back().kind ==
              rund::host::EventKind::IoClose);
  return 0;
}
