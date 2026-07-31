#include "src/host/net/test/socket.hpp"
#include "local.hpp"

#include <sys/socket.h>

bool OpenBasicSyncSockets(BasicSyncSockets &sockets) {
  int native[2] = {-1, -1};
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, native) != 0) {
    return false;
  }
  BasicSocketCleanup left{native[0]};
  BasicSocketCleanup right{native[1]};
  sockets.left = rund::node::test::net::admit(left.fd);
  if (!sockets.left) {
    return false;
  }
  sockets.right = rund::node::test::net::admit(right.fd);
  if (!sockets.right) {
    return false;
  }
  return true;
}
