#include "src/host/net/test/socket.hpp"
#include "local.hpp"

#include "test/assert.hpp"

#include <rund/net/listener.hpp>
#include <sys/socket.h>

int ListenerShutdownConnectedSocketSucceeds() {
  int fds[2] = {-1, -1};
  TEST_ASSERT(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);
  ListenerSocketCloseGuard left{rund::node::test::net::admit(fds[0])};
  ListenerSocketCloseGuard right{rund::node::test::net::admit(fds[1])};

  TEST_ASSERT(rund::node::test::net::native(left.socket) >= 0);
  TEST_ASSERT(rund::node::test::net::native(right.socket) >= 0);

  const auto shutdown =
      rund::net::shutdown(left.socket.view(), rund::net::ShutdownMode::Write);
  TEST_ASSERT(shutdown.ok());
  TEST_ASSERT(shutdown.mode == rund::net::ShutdownMode::Write);

  const auto left_close = left.close();
  TEST_ASSERT(left_close.ok());
  const auto right_close = right.close();
  TEST_ASSERT(right_close.ok());

  return 0;
}
