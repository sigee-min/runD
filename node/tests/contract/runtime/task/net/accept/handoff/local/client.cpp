#include "src/host/net/test/socket.hpp"
#include "../local.hpp"

#include "test/assert.hpp"

#include <rund/net/address.hpp>
#include <rund/net/connection.hpp>
#include <rund/net/socket.hpp>
#include <sys/socket.h>

namespace rund::node::test_contract::net_accept_handoff {

int StartNonblockingClient(const rund::net::Address &connect_address,
                           rund::net::Socket &client) {
  SocketCleanup cleanup{::socket(AF_INET, SOCK_STREAM, 0)};
  TEST_ASSERT(cleanup.fd >= 0);
  client = rund::node::test::net::admit(cleanup.fd);
  TEST_ASSERT(rund::net::nonblocking(client.view(), true).ok());
  const rund::net::connect::Result started =
      rund::net::connect::start(client.view(), connect_address);
  TEST_ASSERT(started.ok());
  return 0;
}

} // namespace rund::node::test_contract::net_accept_handoff
