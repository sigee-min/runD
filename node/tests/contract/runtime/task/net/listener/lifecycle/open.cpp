#include "src/host/net/test/socket.hpp"
#include "local.hpp"

#include "test/assert.hpp"

#include <cstdint>
#include <cstring>
#include <rund/net/address.hpp>
#include <rund/net/listener.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

int ListenerLifecycleOpensBindsListensAndCloses() {
  auto opened = rund::net::open(
      rund::net::OpenOptions{.family = rund::net::Family::IPv4,
                             .transport = rund::net::Transport::Stream,
                             .nonblocking = true});
  TEST_ASSERT(opened.ok());
  TEST_ASSERT(rund::node::test::net::native(opened.socket) >= 0);
  TEST_ASSERT(
      opened.socket.id() ==
      static_cast<std::uint64_t>(rund::node::test::net::native(opened.socket)) +
          1u);
  TEST_ASSERT(rund::node::test::net::generation(opened.socket) != 0u);
  TEST_ASSERT(opened.nonblocking.ok());

  const auto bound =
      rund::net::bind(opened.socket.view(), ListenerLoopbackAnyPort());
  TEST_ASSERT(bound.ok());
  TEST_ASSERT(bound.address_hash.value != 0u);

  const auto listened = rund::net::listen(opened.socket.view(), 64);
  TEST_ASSERT(listened.ok());
  TEST_ASSERT(listened.backlog == 64);

  const auto local = rund::net::local(opened.socket.view());
  TEST_ASSERT(local.ok());
  TEST_ASSERT(local.address.family() == rund::net::Family::IPv4);
  TEST_ASSERT(local.address.port() != 0u);
  TEST_ASSERT(local.address_hash.value != 0u);

  sockaddr_in returned_address{};
  std::memcpy(&returned_address.sin_addr, local.address.bytes().data(),
              local.address.bytes().size());
  TEST_ASSERT(ntohl(returned_address.sin_addr.s_addr) == INADDR_LOOPBACK);

  const auto closed = opened.socket.close();
  TEST_ASSERT(closed.ok());

  return 0;
}
