#include "../local.hpp"
#include <rund/net/socket.hpp>

SocketCloseGuard::SocketCloseGuard(rund::net::Socket opened) noexcept
    : socket(std::move(opened)) {}

[[nodiscard]] rund::net::CloseResult SocketCloseGuard::close() noexcept {
  return socket.close();
}

RawFdGuard::RawFdGuard(const int native) noexcept : fd(native) {}

RawFdGuard::~RawFdGuard() {
  if (fd >= 0) {
    static_cast<void>(::close(fd));
  }
}
