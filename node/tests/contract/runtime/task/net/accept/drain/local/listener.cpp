#include "src/host/net/test/socket.hpp"
#include "../local.hpp"

#include "test/assert.hpp"

#include <cstring>
#include <rund/net/address.hpp>
#include <rund/net/socket.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

namespace {

[[nodiscard]] rund::net::Address
AddressFromSockaddr(const sockaddr_in &address) {
  std::array<std::byte, 4u> bytes{};
  std::memcpy(bytes.data(), &address.sin_addr, bytes.size());
  return rund::net::Address::ipv4(bytes, ntohs(address.sin_port));
}

} // namespace

int PrepareAcceptDrainLoopbackListener(AcceptDrainLoopbackFixture &fixture) {
  const int listener_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  TEST_ASSERT(listener_fd >= 0);
  fixture.listener_cleanup = AcceptDrainSocketCleanup{listener_fd};
  int one = 1;
  TEST_ASSERT(::setsockopt(listener_fd, SOL_SOCKET, SO_REUSEADDR, &one,
                           sizeof(one)) == 0);
  sockaddr_in bind_address{};
  bind_address.sin_family = AF_INET;
  bind_address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  bind_address.sin_port = 0;
  TEST_ASSERT(::bind(listener_fd, reinterpret_cast<sockaddr *>(&bind_address),
                     sizeof(bind_address)) == 0);
  TEST_ASSERT(::listen(listener_fd, 16) == 0);
  sockaddr_in actual_address{};
  socklen_t actual_length = sizeof(actual_address);
  TEST_ASSERT(::getsockname(listener_fd,
                            reinterpret_cast<sockaddr *>(&actual_address),
                            &actual_length) == 0);

  fixture.listener = rund::node::test::net::admit(fixture.listener_cleanup.fd);
  TEST_ASSERT(rund::net::nonblocking(fixture.listener.view(), true).ok());
  fixture.connect_address = AddressFromSockaddr(actual_address);
  return 0;
}
