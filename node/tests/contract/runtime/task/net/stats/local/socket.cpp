#include "../local.hpp"

#include <sys/socket.h>

bool MakeNetStatsSocketPair(NetStatsSocketPairCleanup& cleanup) {
  int fds[2] = {-1, -1};
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
    return false;
  }
  cleanup.left = fds[0];
  cleanup.right = fds[1];
  return true;
}
