#include "src/host/net/test/socket.hpp"
#include "../local.hpp"

#include "test/assert.hpp"

#include <rund/net/socket.hpp>
#include <sys/socket.h>

int PrepareMultiClientLoopbackListener(MultiClientLoopbackFixture &fixture) {
  const int listener_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  TEST_ASSERT(listener_fd >= 0);
  fixture.listener_cleanup.reset(listener_fd);
  int one = 1;
  TEST_ASSERT(::setsockopt(listener_fd, SOL_SOCKET, SO_REUSEADDR, &one,
                           sizeof(one)) == 0);
  sockaddr_in bind_address{};
  bind_address.sin_family = AF_INET;
  bind_address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  bind_address.sin_port = 0;
  TEST_ASSERT(::bind(listener_fd, reinterpret_cast<sockaddr *>(&bind_address),
                     sizeof(bind_address)) == 0);
  TEST_ASSERT(::listen(listener_fd, 128) == 0);
  socklen_t actual_length = sizeof(fixture.connect_address);
  TEST_ASSERT(
      ::getsockname(listener_fd,
                    reinterpret_cast<sockaddr *>(&fixture.connect_address),
                    &actual_length) == 0);

  fixture.listener = rund::node::test::net::admit(fixture.listener_cleanup.fd);
  TEST_ASSERT(rund::net::nonblocking(fixture.listener.view(), true).ok());
  return 0;
}
