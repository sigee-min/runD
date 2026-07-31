#include "src/host/net/test/socket.hpp"
#include "local.hpp"

#include "test/assert.hpp"

#include <rund/net/address.hpp>
#include <rund/net/listener.hpp>
#include <rund/net/socket.hpp>
#include <sys/socket.h>
#include <unistd.h>

int ListenerBacklogAndStaleHandlesFailClosed() {
  auto opened = rund::net::open(
      rund::net::OpenOptions{.family = rund::net::Family::IPv4,
                             .transport = rund::net::Transport::Stream,
                             .nonblocking = true});
  TEST_ASSERT(opened.ok());

  const auto negative_backlog = rund::net::listen(opened.socket.view(), -1);
  TEST_ASSERT(!negative_backlog.ok());
  TEST_ASSERT(negative_backlog.code() == rund::ReasonCode::TaskInvalid);

  int fds[2] = {-1, -1};
  TEST_ASSERT(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);
  rund::net::Socket stale = rund::node::test::net::admit(fds[0]);
  TEST_ASSERT(rund::node::test::net::native(stale) >= 0);
  TEST_ASSERT(stale.close().ok());

  int replacement[2] = {-1, -1};
  TEST_ASSERT(::socketpair(AF_UNIX, SOCK_STREAM, 0, replacement) == 0);

  const auto closed_local = rund::net::local(stale.view());
  TEST_ASSERT(!closed_local.ok());
  TEST_ASSERT(closed_local.code() == rund::ReasonCode::IoFdInvalid);

  const auto stale_bind =
      rund::net::bind(stale.view(), ListenerLoopbackAnyPort());
  TEST_ASSERT(!stale_bind.ok());
  TEST_ASSERT(stale_bind.code() == rund::ReasonCode::IoFdInvalid);

  const auto stale_listen = rund::net::listen(stale.view(), 1);
  TEST_ASSERT(!stale_listen.ok());
  TEST_ASSERT(stale_listen.code() == rund::ReasonCode::IoFdInvalid);

  const auto stale_shutdown =
      rund::net::shutdown(stale.view(), rund::net::ShutdownMode::ReadWrite);
  TEST_ASSERT(!stale_shutdown.ok());
  TEST_ASSERT(stale_shutdown.code() == rund::ReasonCode::IoFdInvalid);

  if (replacement[0] >= 0) {
    static_cast<void>(::close(replacement[0]));
  }
  if (replacement[1] >= 0) {
    static_cast<void>(::close(replacement[1]));
  }
  if (fds[1] >= 0) {
    static_cast<void>(::close(fds[1]));
  }

  return 0;
}
