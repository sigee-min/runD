#include "local.hpp"

#include <sys/socket.h>
#include <unistd.h>

WriteDrainSocketPairCleanup::~WriteDrainSocketPairCleanup() {
  if (left >= 0) {
    static_cast<void>(::close(left));
  }
  if (right >= 0) {
    static_cast<void>(::close(right));
  }
}

bool MakeWriteDrainSocketPair(WriteDrainSocketPairCleanup &cleanup) {
  int fds[2] = {-1, -1};
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
    return false;
  }
  cleanup.left = fds[0];
  cleanup.right = fds[1];
  return true;
}

rund::SessionConfig NetWriteDrainRunSpec() noexcept {
  return rund::SessionConfig{
      .workers = 1u,
      .scheduler =
          {
              .task_capacity = 6u,
              .ready_queue_capacity = 6u,
              .reactor_wait_capacity = 6u,
              .observation_capacity = 128u,
              .host_event_capacity = 128u,
          },
  };
}
