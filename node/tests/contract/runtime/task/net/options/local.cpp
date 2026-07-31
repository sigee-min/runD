#include "local.hpp"
#include <rund/net/address.hpp>
#include <rund/net/listener.hpp>
#include <rund/net/options.hpp>
#include <rund/net/socket.hpp>

NetOptionsSocketCloseGuard::NetOptionsSocketCloseGuard(
    rund::net::Socket opened) noexcept
    : socket(std::move(opened)) {}

bool OpenNetOptionsInetSocket(const rund::net::Transport transport,
                              NetOptionsSocketCloseGuard &guard) {
  rund::net::OpenResult opened = rund::net::open(rund::net::OpenOptions{
      .family = rund::net::Family::IPv4,
      .transport = transport,
      .nonblocking = true,
  });
  NET_OPTIONS_ASSERT(opened.ok());
  guard = NetOptionsSocketCloseGuard{std::move(opened.socket)};
  return true;
}

bool IsTcpNoDelayUdpAcceptedFailure(
    const rund::net::option::Result result) noexcept {
  return !result.ok() && (result.code() == rund::ReasonCode::IoSyscallFailed ||
                          result.code() == rund::ReasonCode::TaskInvalid);
}
