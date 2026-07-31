#include "src/host/net/test/socket.hpp"
#include "local.hpp"

#include <sys/socket.h>

bool OpenSocketPair(SocketPair &pair) {
  int sockets[2] = {-1, -1};
  VECTORED_ASSERT(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
  pair.left = rund::node::test::net::admit(sockets[0]);
  pair.right = rund::node::test::net::admit(sockets[1]);
  VECTORED_ASSERT(rund::node::test::net::native(pair.left) >= 0);
  VECTORED_ASSERT(rund::node::test::net::native(pair.right) >= 0);
  return true;
}

std::string BytesToString(const std::array<std::byte, 6u> &bytes) {
  std::string out{};
  out.reserve(bytes.size());
  for (const std::byte value : bytes) {
    out.push_back(static_cast<char>(value));
  }
  return out;
}
