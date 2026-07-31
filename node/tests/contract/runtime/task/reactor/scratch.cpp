#include "await.hpp"
#include "test/assert.hpp"

#include <rund/host.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include "../../../../../src/runtime/reactor/diagnostics.hpp"

#include <array>
#include <cstddef>

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

} // namespace

int RunRuntimeTaskReactorScratchContract() {
  constexpr std::size_t kTasks = 8u;
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
  std::array<rund::task::IoResult, kTasks> ready{};
  rund::task::Status scoped{};
  bool writes_ok = true;
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 24u,
                  .ready_queue_capacity = 24u,
                  .reactor_wait_capacity = 24u,
                  .observation_capacity = 128u,
                  .host_event_capacity = 128u,
              },
      },
      [&] {
        scoped = rund::task::scope([&] {
          for (std::size_t index = 0u; index < kTasks; ++index) {
            (void)rund::task::spawn(
                "scratch-reader",
                rund::node::test_contract::reactor::AwaitReadable(
                    ready_fds[index].view(), &ready[index]));
          }
          (void)rund::task::spawn("scratch-writer", [&] {
            const char byte = 's';
            for (PipePair &pipe : pipes) {
              writes_ok =
                  (::write(pipe.write_fd, &byte, 1u) == 1) && writes_ok;
            }
          });
        });
      });

  const rund::node::ReactorBackendStats stats =
      rund::node::ReactorBackendStatsSnapshot();
  TEST_ASSERT(report.ok());
  TEST_ASSERT(scoped.ok());
  TEST_ASSERT(writes_ok);
  for (const rund::task::IoResult &result : ready) {
    TEST_ASSERT(result.ok());
  }
  TEST_ASSERT(stats.scratch_ready_reuses > 0u);
  TEST_ASSERT(stats.scratch_host_event_reuses > 0u);
  return 0;
}
