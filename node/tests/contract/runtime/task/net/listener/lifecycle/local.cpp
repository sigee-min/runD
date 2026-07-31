#include "local.hpp"

#include <rund/net/address.hpp>
#include <rund/net/socket.hpp>
#include <utility>

rund::net::Address ListenerLoopbackAnyPort() {
  return rund::net::Address::loopback(rund::net::Family::IPv4);
}

rund::net::Address ListenerLoopbackV6AnyPort() {
  return rund::net::Address::loopback(rund::net::Family::IPv6);
}

ListenerSocketCloseGuard::ListenerSocketCloseGuard(
    rund::net::Socket opened) noexcept
    : socket(std::move(opened)) {}

rund::net::CloseResult ListenerSocketCloseGuard::close() noexcept {
  return socket.close();
}
