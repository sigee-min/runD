#include "../local.hpp"

#include <rund/net/socket.hpp>
#include <unistd.h>
#include <utility>

NetStatsSocketPairCleanup::~NetStatsSocketPairCleanup() {
  if (left >= 0) {
    static_cast<void>(::close(left));
  }
  if (right >= 0) {
    static_cast<void>(::close(right));
  }
}

NetStatsSocketCloseGuard::NetStatsSocketCloseGuard(
    rund::net::Socket opened) noexcept
    : socket(std::move(opened)) {}

rund::net::CloseResult NetStatsSocketCloseGuard::close() noexcept {
  return socket.close();
}
