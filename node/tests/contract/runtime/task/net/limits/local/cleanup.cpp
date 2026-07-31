#include "../local.hpp"

#include <rund/net/socket.hpp>
#include <utility>

namespace rund::node::test_contract::net_limits {

SocketPairCleanup::~SocketPairCleanup() {
  if (left >= 0) {
    static_cast<void>(::close(left));
  }
  if (right >= 0) {
    static_cast<void>(::close(right));
  }
}

NativeSocketCleanup::~NativeSocketCleanup() {
  if (fd >= 0) {
    static_cast<void>(::close(fd));
  }
}

NativeSocketCleanup::NativeSocketCleanup(const int native) noexcept
    : fd(native) {}

void NativeSocketCleanup::release() noexcept { fd = -1; }

SocketCloseGuard::SocketCloseGuard(rund::net::Socket opened) noexcept
    : socket(std::move(opened)) {}

rund::net::CloseResult SocketCloseGuard::close() noexcept {
  return socket.close();
}

} // namespace rund::node::test_contract::net_limits
