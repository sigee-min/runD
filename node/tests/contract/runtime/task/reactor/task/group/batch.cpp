#include "src/host/net/test/socket.hpp"
#include "test/assert.hpp"

#include <rund/net/bytes.hpp>
#include <rund/net/ready.hpp>
#include <rund/net/socket.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include "../../../../../../../src/runtime/reactor/diagnostics.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <thread>
#include <utility>

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

rund::task::Task<void> ReadOne(const rund::net::SocketView socket,
                               std::atomic<std::uint64_t> *const waiting,
                               std::uint64_t *const received) {
  waiting->fetch_add(1u, std::memory_order_acq_rel);
  auto ready = co_await rund::net::ready::read(socket);
  if (!ready.ok()) {
    co_return;
  }
  std::byte byte{};
  const rund::net::ReceiveResult recv =
      rund::net::receive(std::move(ready), std::span<std::byte>{&byte, 1u});
  if (recv.ok() && recv.bytes == 1 && byte == std::byte{0x51}) {
    ++*received;
  }
}

} // namespace

int RunRuntimeTaskReactorTaskGroupBatchContract() {
  constexpr std::size_t kTasks = 128u;
  std::array<SocketPair, kTasks> pairs{};
  std::array<rund::net::Socket, kTasks> left{};
  std::array<rund::net::Socket, kTasks> right{};
  for (std::size_t index = 0u; index < kTasks; ++index) {
    TEST_ASSERT(MakePair(pairs[index]));
    left[index] = rund::node::test::net::admit(pairs[index].left);
    pairs[index].left = -1;
    right[index] = rund::node::test::net::admit(pairs[index].right);
    pairs[index].right = -1;
    TEST_ASSERT(rund::net::nonblocking(left[index].view(), true).ok());
    TEST_ASSERT(rund::net::nonblocking(right[index].view(), true).ok());
  }

  std::atomic<bool> writer_start{false};
  std::atomic<std::uint64_t> waiting_count{0u};
  std::array<std::uint64_t, kTasks> received{};
  bool writer_ok = true;
  std::thread writer([&] {
    while (!writer_start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    while (waiting_count.load(std::memory_order_acquire) < kTasks) {
      std::this_thread::yield();
    }
    const std::byte byte{0x51};
    for (std::size_t index = 0u; index < kTasks; ++index) {
      const rund::net::SendResult sent = rund::net::direct::send(
          left[index].view(), std::span<const std::byte>{&byte, 1u});
      writer_ok = writer_ok && sent.ok() && sent.bytes == 1;
    }
  });

  rund::node::ResetReactorBackendStats();
  rund::task::Status joined{};
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 160u,
                  .ready_queue_capacity = 160u,
                  .reactor_wait_capacity = 160u,
                  .observation_capacity = 512u,
                  .host_event_capacity = 512u,
              },
      },
      [&] {
        writer_start.store(true, std::memory_order_release);
        std::array<rund::task::Handle, kTasks> handles{};
        for (std::size_t index = 0u; index < kTasks; ++index) {
          handles[index] = rund::task::spawn(
              "batch-reader",
              ReadOne(right[index].view(), &waiting_count, &received[index]));
        }
        joined = rund::task::join_all(handles);
      });
  writer.join();

  const rund::node::ReactorBackendStats stats =
      rund::node::ReactorBackendStatsSnapshot();
  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(writer_ok);
  for (const std::uint64_t count : received) {
    TEST_ASSERT(count == 1u);
  }
  TEST_ASSERT(stats.max_registered_fds == kTasks);
  TEST_ASSERT(stats.max_registration_changes_per_apply >= kTasks / 2u);
  TEST_ASSERT(stats.registration_apply_calls <= 8u);
  TEST_ASSERT(stats.registration_apply_deferrals > 0u);
  TEST_ASSERT(stats.registration_apply_deferred_flushes > 0u);
  return 0;
}
