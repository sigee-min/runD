#include "../local.hpp"

#include <unistd.h>

namespace rund::node::test_contract::net_accept_handoff {

SocketCleanup::~SocketCleanup() {
  if (fd >= 0) {
    static_cast<void>(::close(fd));
  }
}

void SocketCleanup::reset(const int native) noexcept {
  if (fd >= 0) {
    static_cast<void>(::close(fd));
  }
  fd = native;
}

}  // namespace rund::node::test_contract::net_accept_handoff
