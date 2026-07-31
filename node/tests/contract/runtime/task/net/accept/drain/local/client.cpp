#include "src/host/net/test/socket.hpp"
#include "../local.hpp"

#include "test/assert.hpp"

#include <cerrno>
#include <rund/net/address.hpp>
#include <rund/net/connection.hpp>
#include <rund/net/socket.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>

namespace {

[[nodiscard]] int StartClient(const rund::net::Address &connect_address,
                              AcceptDrainSocketCleanup &client) {
  client = AcceptDrainSocketCleanup{::socket(AF_INET, SOCK_STREAM, 0)};
  TEST_ASSERT(client.fd >= 0);
  client.socket = rund::node::test::net::admit(client.fd);
  TEST_ASSERT(rund::net::nonblocking(client.socket.view(), true).ok());
  const rund::net::connect::Result started =
      rund::net::connect::start(client.socket.view(), connect_address);
  TEST_ASSERT(started.ok());
  return 0;
}

[[nodiscard]] int WaitClientConnected(const AcceptDrainSocketCleanup &client) {
  const int native = rund::node::test::net::native(client.socket);
  pollfd fd{.fd = native, .events = POLLOUT, .revents = 0};
  int ready = 0;
  do {
    ready = ::poll(&fd, 1u, 1000);
  } while (ready < 0 && errno == EINTR);
  TEST_ASSERT(ready == 1);
  TEST_ASSERT((fd.revents & (POLLOUT | POLLERR | POLLHUP)) != 0);
  int socket_error = 0;
  socklen_t socket_error_size = sizeof(socket_error);
  TEST_ASSERT(::getsockopt(native, SOL_SOCKET, SO_ERROR, &socket_error,
                           &socket_error_size) == 0);
  TEST_ASSERT(socket_error == 0);
  return 0;
}

} // namespace

int StartAcceptDrainClients(const rund::net::Address &connect_address,
                            std::span<AcceptDrainSocketCleanup> clients) {
  for (AcceptDrainSocketCleanup &client : clients) {
    TEST_ASSERT(StartClient(connect_address, client) == 0);
  }
  for (const AcceptDrainSocketCleanup &client : clients) {
    TEST_ASSERT(WaitClientConnected(client) == 0);
  }
  return 0;
}
