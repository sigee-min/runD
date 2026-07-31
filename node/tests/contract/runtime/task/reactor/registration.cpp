#include "test/assert.hpp"

#include <rund/host.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include "../../../../../src/runtime/reactor/diagnostics.hpp"

#include <vector>

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

} // namespace

int RunRuntimeTaskReactorRegistrationContract() {
  PipeCleanup pipe = MakePipe();
  TEST_ASSERT(pipe.read_fd >= 0 && pipe.write_fd >= 0);
  rund::host::io::Fd ready_fd =
      rund::host::io::take_native_fd(::dup(pipe.read_fd));
  TEST_ASSERT(ready_fd);

  rund::node::ResetReactorBackendStats();
  std::vector<int> wake_order{};
  rund::task::IoResult first_ready{};
  rund::task::IoResult second_ready{};
  rund::task::Status joined{};
  bool write_ok = false;

  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 4u,
                  .ready_queue_capacity = 4u,
                  .reactor_wait_capacity = 4u,
                  .observation_capacity = 8u,
                  .host_event_capacity = 8u,
              },
      },
      [&] {
        auto wait_first = [&]() -> rund::task::Task<void> {
          first_ready = co_await rund::host::io::readable(ready_fd.view());
          if (first_ready.ok()) {
            wake_order.push_back(1);
          }
        };
        const rund::task::Handle first =
            rund::task::spawn("reactor-registration-first", wait_first());
        auto wait_second = [&]() -> rund::task::Task<void> {
          second_ready = co_await rund::host::io::readable(ready_fd.view());
          if (second_ready.ok()) {
            wake_order.push_back(2);
          }
        };
        const rund::task::Handle second =
            rund::task::spawn("reactor-registration-second", wait_second());
        const rund::task::Handle writer =
            rund::task::spawn("reactor-registration-writer", [&] {
              const char byte = 'r';
              write_ok = ::write(pipe.write_fd, &byte, 1u) == 1;
            });
        joined = rund::task::join(first, second, writer);
      });

  const rund::node::ReactorBackendStats stats =
      rund::node::ReactorBackendStatsSnapshot();

  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(write_ok);
  TEST_ASSERT(first_ready.ok());
  TEST_ASSERT(second_ready.ok());
  TEST_ASSERT(wake_order.size() == 2u);
  TEST_ASSERT(wake_order[0] == 1);
  TEST_ASSERT(wake_order[1] == 2);
  TEST_ASSERT(stats.add_calls == 1u);
  TEST_ASSERT(stats.modify_calls == 0u);
  TEST_ASSERT(stats.remove_calls <= 1u);
  TEST_ASSERT(stats.deferred_remove_marks == 1u);
  TEST_ASSERT(stats.max_registered_fds == 1u);
  return 0;
}
