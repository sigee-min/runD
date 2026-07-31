#include "src/host/net/test/socket.hpp"
#include "../await.hpp"
#include "test/assert.hpp"

#include <rund/net/bytes.hpp>
#include <rund/net/ready.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/socket.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include <sys/socket.h>
#include <unistd.h>

namespace {

struct SocketPair {
  int left = -1;
  int right = -1;

  ~SocketPair() {
    if (left >= 0) {
      static_cast<void>(::close(left));
    }
    if (right >= 0) {
      static_cast<void>(::close(right));
    }
  }

  SocketPair() = default;
  SocketPair(const SocketPair &) = delete;
  SocketPair &operator=(const SocketPair &) = delete;
};

[[nodiscard]] bool MakePair(SocketPair &pair) {
  int fds[2] = {-1, -1};
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
    return false;
  }
  pair.left = fds[0];
  pair.right = fds[1];
  return true;
}

} // namespace

int RunRuntimeTaskReactorBackendParityContract() {
  constexpr std::size_t kWaits = 8u;
  std::array<SocketPair, kWaits> pairs{};
  std::array<rund::net::Socket, kWaits> left{};
  std::array<rund::net::Socket, kWaits> right{};
  for (std::size_t index = 0u; index < kWaits; ++index) {
    TEST_ASSERT(MakePair(pairs[index]));
    left[index] = rund::node::test::net::admit(pairs[index].left);
    pairs[index].left = -1;
    right[index] = rund::node::test::net::admit(pairs[index].right);
    pairs[index].right = -1;
    TEST_ASSERT(rund::net::nonblocking(left[index].view(), true).ok());
    TEST_ASSERT(rund::net::nonblocking(right[index].view(), true).ok());
  }

  std::array<rund::net::ready::Ticket, kWaits> ready{};
  std::array<std::uint64_t, kWaits> observed{};
  std::uint64_t observed_count = 0u;
  rund::task::Status writer_yield{};
  rund::task::Status joined{};
  bool writer_ok = true;

  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 16u,
                  .ready_queue_capacity = 16u,
                  .reactor_wait_capacity = 16u,
                  .observation_capacity = 64u,
                  .host_event_capacity = 64u,
              },
      },
      [&] {
        std::vector<rund::task::Handle> handles{};
        handles.reserve(kWaits + 1u);
        for (std::size_t index = 0u; index < kWaits; ++index) {
          handles.push_back(rund::task::spawn(
              "reactor-backend-parity-waiter",
              rund::node::test_contract::reactor::AwaitReadable(
                  left[index].view(), &ready[index], [&, index] {
                    if (observed_count < kWaits) {
                      observed[observed_count] =
                          static_cast<std::uint64_t>(index);
                      ++observed_count;
                    }
                  })));
        }
        auto write = [&]() -> rund::task::Task<void> {
          writer_yield = co_await rund::task::yield();
          const std::byte byte{0x70};
          for (std::size_t offset = 0u; offset < kWaits; ++offset) {
            const std::size_t index = kWaits - 1u - offset;
            auto writable =
                co_await rund::net::ready::write(right[index].view());
            const rund::net::SendResult sent = rund::net::send(
                std::move(writable), std::span<const std::byte>{&byte, 1u});
            writer_ok = writer_ok && sent.ok() && sent.bytes == 1;
          }
        };
        handles.push_back(
            rund::task::spawn("reactor-backend-parity-writer", write()));
        joined = rund::task::join_all(handles);
      });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(writer_yield.ok());
  TEST_ASSERT(writer_ok);
  TEST_ASSERT(observed_count == kWaits);
  for (std::size_t index = 0u; index < kWaits; ++index) {
    TEST_ASSERT(ready[index].ok());
    TEST_ASSERT(observed[index] == index);
  }
  TEST_ASSERT(report.tasks().reactor_waits() == kWaits);
  TEST_ASSERT(report.tasks().reactor().ready_events() == kWaits);
  return 0;
}
