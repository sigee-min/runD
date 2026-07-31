#include "src/host/net/test/socket.hpp"
#include "test/assert.hpp"

#include <rund/net/cancel.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/ready/timed.hpp>
#include <rund/net/socket.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>
#include <rund/task/cancel.hpp>

#include <chrono>

#include <sys/socket.h>
#include <unistd.h>

namespace {

struct SocketPairCleanup {
  int left = -1;
  int right = -1;

  ~SocketPairCleanup() {
    if (left >= 0) {
      static_cast<void>(::close(left));
    }
    if (right >= 0) {
      static_cast<void>(::close(right));
    }
  }

  SocketPairCleanup() = default;
  SocketPairCleanup(const SocketPairCleanup &) = delete;
  SocketPairCleanup &operator=(const SocketPairCleanup &) = delete;
};

[[nodiscard]] bool MakeSocketPair(SocketPairCleanup &cleanup) noexcept {
  int fds[2] = {-1, -1};
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
    return false;
  }
  cleanup.left = fds[0];
  cleanup.right = fds[1];
  return true;
}

[[nodiscard]] rund::SessionConfig ReactorCancelCleanupRunSpec() noexcept {
  return rund::SessionConfig{
      .workers = 1u,
      .scheduler =
          {
              .task_capacity = 8u,
              .ready_queue_capacity = 8u,
              .timer_capacity = 8u,
              .reactor_wait_capacity = 8u,
              .observation_capacity = 64u,
              .host_event_capacity = 64u,
          },
  };
}

} // namespace

int RunRuntimeTaskReactorCancelCleanupContract() {
  SocketPairCleanup cleanup{};
  TEST_ASSERT(MakeSocketPair(cleanup));
  rund::net::Socket socket = rund::node::test::net::admit(cleanup.left);
  cleanup.left = -1;
  TEST_ASSERT(rund::net::nonblocking(socket.view(), true).ok());

  rund::net::ready::Ticket ready{};
  rund::task::Status joined{};
  bool source_valid = false;
  bool token_valid = false;
  bool cancel_ok = false;

  const rund::Session::Result report =
      rund::run(ReactorCancelCleanupRunSpec(), [&] {
        auto source = rund::task::stop_source::create();
        source_valid = static_cast<bool>(source);
        if (!source_valid) {
          return;
        }
        auto token = source.token();
        token_valid = static_cast<bool>(token);
        if (!token_valid) {
          return;
        }

        auto wait = [&]() -> rund::task::Task<void> {
          ready = co_await rund::net::ready::timed::read(
              socket.view(), std::chrono::seconds{30}, token);
        };
        const rund::task::Handle waiter =
            rund::task::spawn("reactor-cancel-cleanup-waiter", wait());
        auto cancel = [&]() -> rund::task::Task<void> {
          (void)co_await rund::task::yield();
          cancel_ok = source.request_stop().ok();
        };
        const rund::task::Handle canceller =
            rund::task::spawn("reactor-cancel-cleanup-canceller", cancel());
        joined = rund::task::join(waiter, canceller);
      });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(source_valid);
  TEST_ASSERT(token_valid);
  TEST_ASSERT(cancel_ok);
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(!ready.ok());
  TEST_ASSERT(ready.code() == rund::ReasonCode::TaskCancelled);
  TEST_ASSERT(report.tasks().reactor().waits_canceled() == 1u);
  TEST_ASSERT(report.tasks().reactor().timeout_timer_cancels() == 1u);
  TEST_ASSERT(report.tasks().reactor().timeout_cleanup_failures() == 0u);
  return 0;
}
