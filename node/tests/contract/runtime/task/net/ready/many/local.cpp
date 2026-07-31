#include "local.hpp"

#include <sys/socket.h>
#include <unistd.h>

SocketPairCleanup::~SocketPairCleanup() {
  if (left >= 0) {
    static_cast<void>(::close(left));
  }
  if (right >= 0) {
    static_cast<void>(::close(right));
  }
}

[[nodiscard]] bool MakeSocketPair(SocketPairCleanup &cleanup) {
  int fds[2] = {-1, -1};
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
    return false;
  }
  cleanup.left = fds[0];
  cleanup.right = fds[1];
  return true;
}

[[nodiscard]] rund::SessionConfig
ReadyManyRunSpec(const std::uint32_t task_capacity,
                 const std::uint32_t timer_capacity,
                 const std::uint32_t reactor_wait_capacity) noexcept {
  return rund::SessionConfig{
      .workers = 1u,
      .scheduler =
          {
              .task_workers = 1u,
              .task_capacity = task_capacity,
              .ready_queue_capacity = task_capacity + 4u,
              .timer_capacity = timer_capacity,
              .reactor_wait_capacity = reactor_wait_capacity,
              .observation_capacity = 256u,
              .host_event_capacity = 256u,
          },
  };
}
