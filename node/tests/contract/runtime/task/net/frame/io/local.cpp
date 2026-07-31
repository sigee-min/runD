#include "src/host/net/test/socket.hpp"
#include "local.hpp"

#include <rund/net/socket.hpp>
#include <sys/socket.h>

bool OpenFrameIoNonblockingPair(FrameIoSocketPair &pair) {
  int sockets[2] = {-1, -1};
  FRAMEIO_ASSERT(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
  pair.left = rund::node::test::net::admit(sockets[0]);
  pair.right = rund::node::test::net::admit(sockets[1]);
  FRAMEIO_ASSERT(rund::node::test::net::native(pair.left) >= 0);
  FRAMEIO_ASSERT(rund::node::test::net::native(pair.right) >= 0);
  FRAMEIO_ASSERT(rund::net::nonblocking(pair.left.view(), true).ok());
  FRAMEIO_ASSERT(rund::net::nonblocking(pair.right.view(), true).ok());
  return true;
}

rund::SessionConfig FrameIoRunSpec() noexcept {
  return rund::SessionConfig{
      .workers = 1u,
      .scheduler =
          {
              .task_capacity = 4u,
              .ready_queue_capacity = 4u,
              .reactor_wait_capacity = 4u,
              .observation_capacity = 16u,
              .host_event_capacity = 16u,
          },
  };
}
