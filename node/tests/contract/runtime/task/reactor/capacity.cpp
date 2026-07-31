#include "await.hpp"
#include "test/assert.hpp"

#include <rund/host.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include <string_view>

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

int RunRuntimeTaskReactorCapacityContract() {
  PipeCleanup first = MakePipe();
  PipeCleanup second = MakePipe();
  TEST_ASSERT(first.read_fd >= 0 && first.write_fd >= 0);
  TEST_ASSERT(second.read_fd >= 0 && second.write_fd >= 0);
  rund::host::io::Fd first_fd =
      rund::host::io::take_native_fd(::dup(first.read_fd));
  rund::host::io::Fd second_fd =
      rund::host::io::take_native_fd(::dup(second.read_fd));
  TEST_ASSERT(first_fd && second_fd);
  rund::task::IoResult first_ready{};
  rund::task::IoResult second_ready{};
  rund::task::Status joined{};
  bool wrote_first = false;

  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 4u,
                  .ready_queue_capacity = 4u,
                  .reactor_wait_capacity = 1u,
                  .observation_capacity = 8u,
                  .host_event_capacity = 8u,
              },
      },
      [&] {
        const rund::task::Handle first_reader =
            rund::task::spawn("reactor-capacity-first",
                              rund::node::test_contract::reactor::AwaitReadable(
                                  first_fd.view(), &first_ready));
        const rund::task::Handle second_reader =
            rund::task::spawn("reactor-capacity-second",
                              rund::node::test_contract::reactor::AwaitReadable(
                                  second_fd.view(), &second_ready));
        const rund::task::Handle writer =
            rund::task::spawn("reactor-capacity-writer", [&] {
              const char byte = 'c';
              wrote_first = ::write(first.write_fd, &byte, 1u) == 1;
            });
        joined = rund::task::join(first_reader, second_reader, writer);
      });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(wrote_first);
  TEST_ASSERT(first_ready.ok());
  TEST_ASSERT(!second_ready.ok());
  TEST_ASSERT(second_ready.code() ==
              rund::ReasonCode::ReactorWaitCapacityExceeded);
  TEST_ASSERT(std::string_view{second_ready.error()} ==
              "reactor_wait_capacity_exceeded");
  TEST_ASSERT(report.tasks().reactor_waits() == 1u);

  char discard = 0;
  TEST_ASSERT(::read(first.read_fd, &discard, 1u) == 1);
  TEST_ASSERT(discard == 'c');
  return 0;
}
