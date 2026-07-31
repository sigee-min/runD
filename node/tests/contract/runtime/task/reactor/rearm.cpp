#include "test/assert.hpp"

#include <rund/host.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include "../../../../../src/runtime/reactor/diagnostics.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <thread>

#include <unistd.h>

namespace {

struct PipePair {
  int read_fd = -1;
  int write_fd = -1;

  ~PipePair() {
    if (read_fd >= 0) {
      static_cast<void>(::close(read_fd));
    }
    if (write_fd >= 0) {
      static_cast<void>(::close(write_fd));
    }
  }
};

[[nodiscard]] bool MakePipe(PipePair &pipe) {
  int fds[2] = {-1, -1};
  if (::pipe(fds) != 0) {
    return false;
  }
  pipe.read_fd = fds[0];
  pipe.write_fd = fds[1];
  return true;
}

[[nodiscard]] bool
AllReadsAtLeast(const std::array<std::atomic<std::uint64_t>, 8u> &reads,
                const std::uint64_t count) noexcept {
  for (const std::atomic<std::uint64_t> &item : reads) {
    if (item.load(std::memory_order_acquire) < count) {
      return false;
    }
  }
  return true;
}

rund::task::Task<void> ReadRounds(PipePair *const pipe,
                                  const rund::host::io::FdView ready_fd,
                                  std::atomic<std::uint64_t> *const reads,
                                  const std::size_t iterations) {
  for (std::size_t round = 0u; round < iterations; ++round) {
    const rund::task::IoResult ready =
        co_await rund::host::io::readable(ready_fd);
    if (!ready.ok()) {
      co_return;
    }
    char byte = 0;
    if (::read(pipe->read_fd, &byte, 1u) == 1) {
      reads->fetch_add(1u, std::memory_order_acq_rel);
    }
  }
}

} // namespace

int RunRuntimeTaskReactorRearmContract() {
  constexpr std::size_t kTasks = 8u;
  constexpr std::size_t kIterations = 8u;
  std::array<PipePair, kTasks> pipes{};
  for (PipePair &pipe : pipes) {
    TEST_ASSERT(MakePipe(pipe));
  }
  std::array<rund::host::io::Fd, kTasks> ready_fds{};
  for (std::size_t index = 0u; index < kTasks; ++index) {
    ready_fds[index] =
        rund::host::io::take_native_fd(::dup(pipes[index].read_fd));
    TEST_ASSERT(ready_fds[index]);
  }

  rund::node::ResetReactorBackendStats();
  std::array<std::atomic<std::uint64_t>, kTasks> reads{};
  for (std::atomic<std::uint64_t> &count : reads) {
    count.store(0u, std::memory_order_relaxed);
  }
  std::atomic<bool> writes_ok{true};
  std::thread writer{[&] {
    const char byte = 'r';
    for (std::size_t round = 0u; round < kIterations; ++round) {
      for (PipePair &pipe : pipes) {
        if (::write(pipe.write_fd, &byte, 1u) != 1) {
          writes_ok.store(false, std::memory_order_release);
        }
      }
      while (!AllReadsAtLeast(reads, round + 1u)) {
        std::this_thread::yield();
      }
    }
  }};
  rund::task::Status joined{};
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 32u,
                  .ready_queue_capacity = 32u,
                  .reactor_wait_capacity = 32u,
                  .observation_capacity = 512u,
                  .host_event_capacity = 512u,
              },
      },
      [&] {
        std::array<rund::task::Handle, kTasks> handles{};
        for (std::size_t index = 0u; index < kTasks; ++index) {
          handles[index] = rund::task::spawn(
              "reactor-rearm-reader",
              ReadRounds(&pipes[index], ready_fds[index].view(), &reads[index],
                         kIterations));
        }
        joined = rund::task::join_all(handles);
      });
  writer.join();

  const rund::node::ReactorBackendStats stats =
      rund::node::ReactorBackendStatsSnapshot();
  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(writes_ok.load(std::memory_order_acquire));
  for (const std::atomic<std::uint64_t> &count : reads) {
    TEST_ASSERT(count.load(std::memory_order_acquire) == kIterations);
  }
  TEST_ASSERT(stats.max_registered_fds == kTasks);
  TEST_ASSERT(stats.add_calls <= kTasks * 2u);
  TEST_ASSERT(stats.remove_calls <= kTasks * 2u);
  TEST_ASSERT(stats.deferred_remove_cancellations >= kTasks);
  return 0;
}
