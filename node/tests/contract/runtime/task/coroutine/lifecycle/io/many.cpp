#include "src/host/net/test/socket.hpp"
#include "../../local.hpp"
#include "ops.hpp"
#include "src/host/net/test/socket.hpp"
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>

#include <array>

#include <sys/socket.h>

namespace rund::node::test_contract::coroutine {

int CheckCoroutineReadyMany() {
  int pipe_fds[2] = {-1, -1};
  TEST_ASSERT(::socketpair(AF_UNIX, SOCK_STREAM, 0, pipe_fds) == 0);
  std::atomic<std::uint32_t> after{0u};
  std::uint32_t events = 0u;
  rund::task::Status joined{};
  bool handle_valid = false;
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .id = 790u,
          .workers = 2u,
          .scheduler =
              {
                  .task_capacity = 4u,
                  .ready_queue_capacity = 4u,
              },
      },
      [&] {
        rund::net::Socket socket = rund::node::test::net::admit(pipe_fds[0]);
        pipe_fds[0] = -1;
        const rund::task::Handle reader =
            rund::task::spawn("coroutine-ready-many",
                              ReadyManyAwait(socket.view(), &after, &events));
        const rund::task::Handle writer = rund::task::spawn(
            "coroutine-ready-many-writer", WriteAfterYield(pipe_fds[1]));
        handle_valid = static_cast<bool>(reader) && static_cast<bool>(writer);
        joined = rund::task::join(reader, writer);
      });
  TEST_ASSERT(pipe_fds[0] < 0);
  TEST_ASSERT(::close(pipe_fds[1]) == 0);
  if (AssertCoroutineAwaitSuccess(report, joined, handle_valid, after, 2u,
                                  2u) != 0) {
    return 1;
  }
  TEST_ASSERT(events == 1u);
  return 0;
}

int CheckDiscardedReadyOps() {
  int pipe_fds[2] = {-1, -1};
  TEST_ASSERT(::socketpair(AF_UNIX, SOCK_STREAM, 0, pipe_fds) == 0);
  std::atomic<std::uint32_t> completed{0u};
  rund::task::Status joined{};
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .id = 796u,
          .workers = 2u,
          .scheduler =
              {
                  .task_capacity = 2u,
                  .ready_queue_capacity = 2u,
              },
      },
      [&] {
        rund::net::Socket socket = rund::node::test::net::admit(pipe_fds[0]);
        pipe_fds[0] = -1;
        const rund::task::Handle task =
            rund::task::spawn("coroutine-ready-discard",
                              DiscardReadyOps(socket.view(), &completed));
        joined = rund::task::join(task);
      });
  TEST_ASSERT(pipe_fds[0] < 0);
  TEST_ASSERT(::close(pipe_fds[1]) == 0);
  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(completed.load(std::memory_order_acquire) == 1u);
  TEST_ASSERT(report.tasks().reactor_waits() == 0u);
  TEST_ASSERT(report.tasks().timers() == 0u);
  TEST_ASSERT(report.tasks().parked() == 0u);
  return 0;
}

} // namespace rund::node::test_contract::coroutine
