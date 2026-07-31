#include "../local.hpp"
#include "io/ops.hpp"
#include <rund/task/api.hpp>

#include <array>

namespace rund::node::test_contract::coroutine {

int CheckCoroutineReadyIo() {
  int pipe_fds[2] = {-1, -1};
  TEST_ASSERT(::pipe(pipe_fds) == 0);
  rund::host::io::Fd ready_fd =
      rund::host::io::take_native_fd(::dup(pipe_fds[0]));
  TEST_ASSERT(ready_fd);
  const char ready_byte = 'x';
  TEST_ASSERT(::write(pipe_fds[1], &ready_byte, 1u) == 1);
  std::atomic<std::uint32_t> after{0u};
  short revents = 0;
  rund::task::Status joined{};
  bool handle_valid = false;
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .id = 788u,
          .workers = 2u,
          .scheduler =
              {
                  .task_capacity = 4u,
                  .ready_queue_capacity = 4u,
              },
      },
      [&] {
        const rund::task::Handle task =
            rund::task::spawn("coroutine-ready-io",
                              ReadyIoAwait(ready_fd.view(), &after, &revents));
        handle_valid = static_cast<bool>(task);
        joined = rund::task::join(task);
      });
  TEST_ASSERT(::close(pipe_fds[0]) == 0);
  TEST_ASSERT(::close(pipe_fds[1]) == 0);
  if (AssertCoroutineAwaitSuccess(report, joined, handle_valid, after, 1u,
                                  1u) != 0) {
    return 1;
  }
  TEST_ASSERT(revents != 0);
  return 0;
}

int CheckCoroutineBlockedIo() {
  int pipe_fds[2] = {-1, -1};
  TEST_ASSERT(::pipe(pipe_fds) == 0);
  rund::host::io::Fd ready_fd =
      rund::host::io::take_native_fd(::dup(pipe_fds[0]));
  TEST_ASSERT(ready_fd);
  std::atomic<std::uint32_t> after{0u};
  short revents = 0;
  rund::task::Status joined{};
  bool handle_valid = false;
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .id = 789u,
          .workers = 2u,
          .scheduler =
              {
                  .task_capacity = 4u,
                  .ready_queue_capacity = 4u,
              },
      },
      [&] {
        const rund::task::Handle reader = rund::task::spawn(
            "coroutine-blocked-io",
            BlockedIoAwait(ready_fd.view(), &after, &revents));
        const rund::task::Handle writer = rund::task::spawn(
            "coroutine-blocked-io-writer", WriteAfterYield(pipe_fds[1]));
        handle_valid = static_cast<bool>(reader) && static_cast<bool>(writer);
        joined = rund::task::join(reader, writer);
      });
  TEST_ASSERT(::close(pipe_fds[0]) == 0);
  TEST_ASSERT(::close(pipe_fds[1]) == 0);
  if (AssertCoroutineAwaitSuccess(report, joined, handle_valid, after, 2u,
                                  2u) != 0) {
    return 1;
  }
  TEST_ASSERT(revents != 0);
  return 0;
}

} // namespace rund::node::test_contract::coroutine
