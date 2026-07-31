#include "../local.hpp"

namespace rund::node::test_contract::net_limits {

bool MakeLoopbackListener(int* const out_fd, sockaddr_in* const out_address) {
  if (out_fd == nullptr || out_address == nullptr) {
    return false;
  }
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    return false;
  }
  NativeSocketCleanup cleanup{fd};
  int reuse = 1;
  static_cast<void>(
      ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)));
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  if (::bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
    return false;
  }
  socklen_t length = sizeof(address);
  if (::getsockname(fd, reinterpret_cast<sockaddr*>(&address), &length) != 0) {
    return false;
  }
  if (::listen(fd, 4) != 0) {
    return false;
  }
  *out_fd = fd;
  *out_address = address;
  cleanup.release();
  return true;
}

}  // namespace rund::node::test_contract::net_limits
