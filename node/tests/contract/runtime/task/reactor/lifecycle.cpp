#include "await.hpp"
#include "test/assert.hpp"

#include <rund/host.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include "../../../../../src/runtime/reactor/diagnostics.hpp"

#include <unistd.h>

namespace {

struct PipeCleanup {
  int read_fd = -1;
  int write_fd = -1;

  ~PipeCleanup() {
    if (read_fd >= 0) {
      static_cast<void>(::close(read_fd));
    }
    if (write_fd >= 0) {
      static_cast<void>(::close(write_fd));
    }
  }

  PipeCleanup() = default;
  PipeCleanup(const PipeCleanup &) = delete;
  PipeCleanup &operator=(const PipeCleanup &) = delete;
  PipeCleanup(PipeCleanup &&other) noexcept
      : read_fd(other.read_fd), write_fd(other.write_fd) {
    other.read_fd = -1;
    other.write_fd = -1;
  }
  PipeCleanup &operator=(PipeCleanup &&other) noexcept {
    if (this == &other) {
      return *this;
    }
    this->~PipeCleanup();
    read_fd = other.read_fd;
    write_fd = other.write_fd;
    other.read_fd = -1;
    other.write_fd = -1;
    return *this;
  }
};

[[nodiscard]] PipeCleanup MakePipe() {
  int fds[2] = {-1, -1};
  if (::pipe(fds) != 0) {
    return {};
  }
  PipeCleanup pipe{};
  pipe.read_fd = fds[0];
  pipe.write_fd = fds[1];
  return pipe;
}

[[nodiscard]] rund::Session::Result RunOneReadyWait() {
  PipeCleanup pipe = MakePipe();
  if (pipe.read_fd < 0 || pipe.write_fd < 0) {
    return {};
  }
  rund::host::io::Fd ready_fd =
      rund::host::io::take_native_fd(::dup(pipe.read_fd));
  if (!ready_fd) {
    return {};
  }
  rund::task::IoResult ready{};
  rund::task::Status joined{};
  bool write_ok = false;
  return rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 3u,
                  .ready_queue_capacity = 3u,
                  .reactor_wait_capacity = 3u,
                  .observation_capacity = 4u,
                  .host_event_capacity = 4u,
              },
      },
      [&] {
        const rund::task::Handle reader =
            rund::task::spawn("reactor-lifecycle-reader",
                              rund::node::test_contract::reactor::AwaitReadable(
                                  ready_fd.view(), &ready));
        const rund::task::Handle writer =
            rund::task::spawn("reactor-lifecycle-writer", [&] {
              const char byte = 'l';
              write_ok = ::write(pipe.write_fd, &byte, 1u) == 1;
            });
        joined = rund::task::join(reader, writer);
        if (!ready.ok() || !joined.ok() || !write_ok) {
          return;
        }
      });
}

} // namespace

int RunRuntimeTaskReactorLifecycleContract() {
  rund::node::ResetReactorBackendStats();

  const rund::Session::Result first = RunOneReadyWait();
  TEST_ASSERT(first.ok());
  const rund::Session::Result second = RunOneReadyWait();
  TEST_ASSERT(second.ok());

  const rund::node::ReactorBackendStats stats =
      rund::node::ReactorBackendStatsSnapshot();
  TEST_ASSERT(stats.open_calls == 2u);
  TEST_ASSERT(stats.close_calls == 2u);
  TEST_ASSERT(stats.max_open_handles == 1u);
  TEST_ASSERT(stats.current_open_handles == 0u);
  TEST_ASSERT(stats.current_registered_fds == 0u);
  return 0;
}
