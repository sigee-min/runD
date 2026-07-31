#pragma once

#include <rund/net/address.hpp>
#include <rund/net/socket.hpp>
#include <rund/session.hpp>

#include <netinet/in.h>

namespace rund::node::test_contract::net_accept_handoff {

struct SocketCleanup {
  int fd = -1;

  ~SocketCleanup();
  SocketCleanup() = default;
  explicit SocketCleanup(const int native) noexcept : fd(native) {}
  SocketCleanup(const SocketCleanup &) = delete;
  SocketCleanup &operator=(const SocketCleanup &) = delete;

  void reset(int native) noexcept;
};

struct LoopbackFixture {
  SocketCleanup listener_cleanup{};
  rund::net::Socket listener{};
  rund::net::Address connect_address{};
};

rund::net::Address AddressFromSockaddr(const sockaddr_in &address);
int PrepareLoopbackListener(LoopbackFixture &fixture);
int StartNonblockingClient(const rund::net::Address &connect_address,
                           rund::net::Socket &client);
rund::SessionConfig RunSpec() noexcept;

} // namespace rund::node::test_contract::net_accept_handoff

int RunNetAcceptHandoffPreparedSocketCase();
