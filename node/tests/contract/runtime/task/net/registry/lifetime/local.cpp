#include "local.hpp"

#include <utility>

#include <sys/socket.h>
#include <unistd.h>

namespace rund::node::test_contract::net_registry_lifetime {

SocketPair::~SocketPair() {
  Close(left);
  Close(right);
}

void SocketPair::Close(int& fd) noexcept {
  if (fd >= 0) {
    static_cast<void>(::close(fd));
    fd = -1;
  }
}

bool MakeSocketPair(SocketPair& pair) noexcept {
  SocketPair::Close(pair.left);
  SocketPair::Close(pair.right);
  int fds[2] = {-1, -1};
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
    return false;
  }
  pair.left = fds[0];
  pair.right = fds[1];
  return true;
}

bool ForceLeftFd(SocketPair& pair, const int fd) noexcept {
  if (pair.right == fd) {
    std::swap(pair.left, pair.right);
  }
  if (pair.left == fd) {
    return true;
  }
  if (::dup2(pair.left, fd) != fd) {
    return false;
  }
  SocketPair::Close(pair.left);
  pair.left = fd;
  return true;
}

}  // namespace rund::node::test_contract::net_registry_lifetime
