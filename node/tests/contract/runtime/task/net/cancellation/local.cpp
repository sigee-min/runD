#include "local.hpp"

#include <array>
#include <cerrno>
#include <cstddef>

#include <sys/socket.h>
#include <unistd.h>

CancelSocketPairCleanup::~CancelSocketPairCleanup() {
  if (left >= 0) {
    static_cast<void>(::close(left));
  }
  if (right >= 0) {
    static_cast<void>(::close(right));
  }
}

bool MakeCancelSocketPair(CancelSocketPairCleanup &cleanup) {
  int fds[2] = {-1, -1};
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
    return false;
  }
  cleanup.left = fds[0];
  cleanup.right = fds[1];
  return true;
}

bool SaturateSocket(const int fd) {
  std::array<std::byte, 4096u> bytes{};
  for (;;) {
    const ssize_t sent = ::send(fd, bytes.data(), bytes.size(), 0);
    if (sent > 0) {
      continue;
    }
    if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      return true;
    }
    return false;
  }
}

rund::SessionConfig NetCancellationRunSpec() noexcept {
  return rund::SessionConfig{
      .workers = 1u,
      .scheduler =
          {
              .task_workers = 1u,
              .task_capacity = 8u,
              .ready_queue_capacity = 8u,
              .timer_capacity = 8u,
              .reactor_wait_capacity = 8u,
              .observation_capacity = 128u,
              .host_event_capacity = 128u,
          },
  };
}
