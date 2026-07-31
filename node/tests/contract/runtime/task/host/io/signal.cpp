#include "test/assert.hpp"

#include "local.hpp"

#include "../../../../../../src/host/io/access.hpp"

#include <rund/host.hpp>
#include <rund/host/io.hpp>
#include <rund/replay.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include <array>
#include <cstddef>
#include <span>

#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>

namespace runtime_task_host_io {

void Signal() {
  struct sigaction default_action {};
  struct sigaction previous_action {};
  default_action.sa_handler = SIG_DFL;
  TEST_ASSERT(sigemptyset(&default_action.sa_mask) == 0);
  TEST_ASSERT(::sigaction(SIGPIPE, &default_action, &previous_action) == 0);

  int pipe_fds[2] = {-1, -1};
  TEST_ASSERT(::pipe(pipe_fds) == 0);
  Fd read_cleanup{pipe_fds[0]};
  Fd write_cleanup{pipe_fds[1]};
  TEST_ASSERT(::close(read_cleanup.native) == 0);
  read_cleanup.release();
  rund::host::io::Fd fd =
      rund::host::io::take_native_fd(write_cleanup.native);
  const std::array<std::byte, 1u> payload{std::byte{0x7e}};
#if defined(F_GETNOSIGPIPE)
  TEST_ASSERT(::fcntl(rund::host::io::detail::Access::native(fd.view()),
                      F_GETNOSIGPIPE) == 0);
#endif

  const rund::SessionConfig config{
      .workers = 1u,
      .scheduler =
          {
              .task_workers = 1u,
              .task_capacity = 2u,
              .ready_queue_capacity = 2u,
              .host_io_capacity = 1u,
              .host_event_capacity = 1u,
              .host_payload_capacity_bytes = 1u,
          },
  };
  rund::host::io::WriteResult record_write{};
  rund::task::Status record_join{};
  rund::Session session{};
  TEST_ASSERT(session.open(config));
  const rund::replay::Record recorded =
      rund::replay::record(session, [&](rund::replay::Context &) {
        auto body = [&]() -> rund::task::Task<void> {
          record_write = co_await rund::host::io::write_some(
              fd.view(), std::span<const std::byte>{payload});
        };
        const rund::task::Handle task =
            rund::task::spawn("hostio-epipe", body());
        record_join = rund::task::join(task);
      });
  TEST_ASSERT(recorded);
  TEST_ASSERT(record_join);
  TEST_ASSERT(!record_write);
  TEST_ASSERT(record_write.code() == rund::ReasonCode::IoSyscallFailed);
  TEST_ASSERT(record_write.native_error == EPIPE);
  TEST_ASSERT(recorded.host_event_count() == 1u);
  TEST_ASSERT(recorded.storage_report().logical_bytes == 0u);
#if defined(F_GETNOSIGPIPE)
  TEST_ASSERT(::fcntl(rund::host::io::detail::Access::native(fd.view()),
                      F_GETNOSIGPIPE) == 1);
#endif
  struct sigaction current_action {};
  TEST_ASSERT(::sigaction(SIGPIPE, nullptr, &current_action) == 0);
  TEST_ASSERT(current_action.sa_handler == SIG_DFL);

  const rund::host::io::WriteResult blocking =
      rund::host::io::write_some_blocking(fd.view(),
                                          std::span<const std::byte>{payload});
  TEST_ASSERT(!blocking);
  TEST_ASSERT(blocking.code() == rund::ReasonCode::IoSyscallFailed);
  TEST_ASSERT(blocking.native_error == EPIPE);

  rund::host::io::WriteResult replay_write{};
  rund::task::Status replay_join{};
  rund::host::io::Fd replay_fd = rund::host::io::replay_fd(404u);
  const rund::replay::Check replayed =
      rund::replay::run(session, recorded, [&](rund::replay::Context &) {
        auto body = [&]() -> rund::task::Task<void> {
          replay_write = co_await rund::host::io::write_some(
              replay_fd.view(), std::span<const std::byte>{payload});
        };
        const rund::task::Handle task =
            rund::task::spawn("hostio-epipe", body());
        replay_join = rund::task::join(task);
      });
  TEST_ASSERT(replayed);
  TEST_ASSERT(replay_join);
  TEST_ASSERT(!replay_write);
  TEST_ASSERT(replay_write.code() == rund::ReasonCode::IoSyscallFailed);
  TEST_ASSERT(replay_write.native_error == EPIPE);
  TEST_ASSERT(replayed.actual().has_value());
  TEST_ASSERT(replayed.actual()->tasks().external_parks() == 0u);
  TEST_ASSERT(replayed.actual()->tasks().external_wakes() == 0u);
  TEST_ASSERT(session.close());

  sigset_t pipe_mask{};
  sigset_t previous_mask{};
  TEST_ASSERT(sigemptyset(&pipe_mask) == 0);
  TEST_ASSERT(sigaddset(&pipe_mask, SIGPIPE) == 0);
  TEST_ASSERT(::pthread_sigmask(SIG_BLOCK, &pipe_mask, &previous_mask) == 0);
  sigset_t pending{};
  TEST_ASSERT(::sigpending(&pending) == 0);
  TEST_ASSERT(sigismember(&pending, SIGPIPE) == 0);
  const rund::host::io::WriteResult blocked_write =
      rund::host::io::write_some_blocking(fd.view(),
                                          std::span<const std::byte>{payload});
  TEST_ASSERT(!blocked_write);
  TEST_ASSERT(blocked_write.native_error == EPIPE);
  TEST_ASSERT(::sigpending(&pending) == 0);
  TEST_ASSERT(sigismember(&pending, SIGPIPE) == 0);

  TEST_ASSERT(::raise(SIGPIPE) == 0);
  TEST_ASSERT(::sigpending(&pending) == 0);
  TEST_ASSERT(sigismember(&pending, SIGPIPE) == 1);
  const rund::host::io::WriteResult pending_write =
      rund::host::io::write_some_blocking(fd.view(),
                                          std::span<const std::byte>{payload});
  TEST_ASSERT(!pending_write);
  TEST_ASSERT(pending_write.native_error == EPIPE);
  TEST_ASSERT(::sigpending(&pending) == 0);
  TEST_ASSERT(sigismember(&pending, SIGPIPE) == 1);
  int consumed_signal = 0;
  TEST_ASSERT(::sigwait(&pipe_mask, &consumed_signal) == 0);
  TEST_ASSERT(consumed_signal == SIGPIPE);
  TEST_ASSERT(::sigpending(&pending) == 0);
  TEST_ASSERT(sigismember(&pending, SIGPIPE) == 0);
  TEST_ASSERT(::pthread_sigmask(SIG_SETMASK, &previous_mask, nullptr) == 0);
  TEST_ASSERT(::sigaction(SIGPIPE, &previous_action, nullptr) == 0);
}

} // namespace runtime_task_host_io
