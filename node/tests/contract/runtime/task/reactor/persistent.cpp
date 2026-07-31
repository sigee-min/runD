#include "test/assert.hpp"

#include <rund/host.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include "../../../../../src/runtime/reactor/diagnostics.hpp"

#include <array>

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

int RunRuntimeTaskReactorPersistentContract() {
  PipeCleanup pipe = MakePipe();
  TEST_ASSERT(pipe.read_fd >= 0 && pipe.write_fd >= 0);
  rund::host::io::Fd ready_fd =
      rund::host::io::take_native_fd(::dup(pipe.read_fd));
  TEST_ASSERT(ready_fd);

  rund::node::ResetReactorBackendStats();
  std::array<char, 2u> bytes{};
  rund::task::IoResult first_ready{};
  rund::task::IoResult second_ready{};
  rund::task::Status yielded{};
  rund::task::Status joined{};
  bool read_all = false;
  bool wrote_all = false;

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
        auto read = [&]() -> rund::task::Task<void> {
          first_ready = co_await rund::host::io::readable(ready_fd.view());
          if (!first_ready.ok() ||
              ::read(pipe.read_fd, &bytes[0], 1u) != 1) {
            co_return;
          }
          second_ready = co_await rund::host::io::readable(ready_fd.view());
          if (!second_ready.ok()) {
            co_return;
          }
          read_all = ::read(pipe.read_fd, &bytes[1], 1u) == 1;
        };
        const rund::task::Handle reader =
            rund::task::spawn("reactor-persistent-reader", read());
        auto write = [&]() -> rund::task::Task<void> {
          const char first = 'a';
          const char second = 'b';
          if (::write(pipe.write_fd, &first, 1u) != 1) {
            co_return;
          }
          yielded = co_await rund::task::yield();
          wrote_all = ::write(pipe.write_fd, &second, 1u) == 1;
        };
        const rund::task::Handle writer =
            rund::task::spawn("reactor-persistent-writer", write());
        joined = rund::task::join(reader, writer);
      });

  const rund::node::ReactorBackendStats stats =
      rund::node::ReactorBackendStatsSnapshot();

  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(read_all);
  TEST_ASSERT(wrote_all);
  TEST_ASSERT(yielded.ok());
  TEST_ASSERT(first_ready.ok());
  TEST_ASSERT(second_ready.ok());
  TEST_ASSERT(bytes[0] == 'a');
  TEST_ASSERT(bytes[1] == 'b');
  TEST_ASSERT(stats.open_calls == 1u);
  TEST_ASSERT(stats.close_calls == 1u);
  TEST_ASSERT(stats.max_open_handles == 1u);
  TEST_ASSERT(stats.poll_calls >= 1u);
  TEST_ASSERT(stats.add_calls >= 1u);
  return 0;
}
