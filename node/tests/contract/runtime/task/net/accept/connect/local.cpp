#include "local.hpp"

#include <cstring>
#include <rund/net/address.hpp>

#include <sys/socket.h>
#include <unistd.h>

AcceptConnectSocketCleanup::~AcceptConnectSocketCleanup() {
  if (fd >= 0) {
    static_cast<void>(::close(fd));
  }
}

AcceptConnectSocketCleanup::AcceptConnectSocketCleanup(
    const int native) noexcept
    : fd(native) {}

void AcceptConnectSocketCleanup::reset(const int native) noexcept {
  if (fd >= 0) {
    static_cast<void>(::close(fd));
  }
  fd = native;
}

void AcceptConnectSocketCleanup::release() noexcept { fd = -1; }

rund::net::Address
AcceptConnectAddressFromSockaddr(const sockaddr_in &address) {
  std::array<std::byte, 4u> bytes{};
  std::memcpy(bytes.data(), &address.sin_addr, bytes.size());
  return rund::net::Address::ipv4(bytes, ntohs(address.sin_port));
}

bool MakeAcceptConnectLoopbackListener(int *const out_fd,
                                       sockaddr_in *const out_address) {
  if (out_fd == nullptr || out_address == nullptr) {
    return false;
  }
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    return false;
  }
  AcceptConnectSocketCleanup cleanup{fd};
  int reuse = 1;
  static_cast<void>(
      ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)));
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  if (::bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) !=
      0) {
    return false;
  }
  socklen_t length = sizeof(address);
  if (::getsockname(fd, reinterpret_cast<sockaddr *>(&address), &length) != 0) {
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
