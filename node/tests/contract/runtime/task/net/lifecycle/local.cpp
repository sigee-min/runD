#include "local.hpp"

#include <utility>

#include <sys/socket.h>
#include <unistd.h>

namespace rund::node::test_contract::net_lifecycle {

SocketPairCleanup::~SocketPairCleanup() {
  if (left >= 0) {
    static_cast<void>(::close(left));
  }
  if (right >= 0) {
    static_cast<void>(::close(right));
  }
}

bool MakeSocketPair(SocketPairCleanup &cleanup) {
  int fds[2] = {-1, -1};
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
    return false;
  }
  cleanup.left = fds[0];
  cleanup.right = fds[1];
  return true;
}

bool ForceLeftFd(SocketPairCleanup &cleanup, const int fd) noexcept {
  if (cleanup.right == fd) {
    std::swap(cleanup.left, cleanup.right);
  }
  if (cleanup.left == fd) {
    return true;
  }
  if (::dup2(cleanup.left, fd) != fd) {
    return false;
  }
  static_cast<void>(::close(cleanup.left));
  cleanup.left = fd;
  return true;
}

rund::SessionConfig RunSpec() noexcept {
  return rund::SessionConfig{
      .workers = 1u,
      .scheduler =
          {
              .task_capacity = 4u,
              .ready_queue_capacity = 4u,
              .reactor_wait_capacity = 4u,
              .observation_capacity = 32u,
              .host_event_capacity = 32u,
          },
  };
}

} // namespace rund::node::test_contract::net_lifecycle
