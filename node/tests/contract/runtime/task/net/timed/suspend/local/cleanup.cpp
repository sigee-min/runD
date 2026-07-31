#include "../local.hpp"

#include <sys/socket.h>
#include <unistd.h>

TimedSuspendSocketPairCleanup::~TimedSuspendSocketPairCleanup() {
  if (left >= 0) {
    static_cast<void>(::close(left));
  }
  if (right >= 0) {
    static_cast<void>(::close(right));
  }
}

bool MakeTimedSuspendSocketPair(TimedSuspendSocketPairCleanup& cleanup) {
  int fds[2] = {-1, -1};
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
    return false;
  }
  cleanup.left = fds[0];
  cleanup.right = fds[1];
  return true;
}
