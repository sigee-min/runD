#include "src/host/net/test/ticket.hpp"
#include "src/host/net/test/socket.hpp"
#include "local.hpp"

#include "test/assert.hpp"

#include <array>
#include <cstddef>
#include <rund/net/address.hpp>
#include <rund/net/bytes.hpp>
#include <rund/net/connection.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/socket.hpp>
#include <span>
#include <utility>

#include <sys/socket.h>
#include <unistd.h>

int RunAcceptConnectBasicCase() {
  int listener_fd = -1;
  sockaddr_in listener_address{};
  TEST_ASSERT(
      MakeAcceptConnectLoopbackListener(&listener_fd, &listener_address));
  AcceptConnectSocketCleanup listener_cleanup{listener_fd};
  rund::net::Socket listener =
      rund::node::test::net::admit(listener_cleanup.fd);
  TEST_ASSERT(rund::net::nonblocking(listener.view(), true).ok());

  const rund::net::accept::Result empty_accept =
      rund::net::accept::one(rund::node::test::net::ticket(
          listener.view(), rund::net::ready::Interest::Readable));
  TEST_ASSERT(!empty_accept.ok());
  TEST_ASSERT(empty_accept.code() == rund::ReasonCode::IoWouldBlock);

  const int client_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  TEST_ASSERT(client_fd >= 0);
  AcceptConnectSocketCleanup client_cleanup{client_fd};
  rund::net::Socket client = rund::node::test::net::admit(client_cleanup.fd);
  TEST_ASSERT(rund::net::nonblocking(client.view(), true).ok());

  const rund::net::Address connect_address =
      AcceptConnectAddressFromSockaddr(listener_address);
  const rund::net::connect::Result started =
      rund::net::connect::start(client.view(), connect_address);
  TEST_ASSERT(started.ok());

  rund::net::accept::Result accepted{};
  for (int attempt = 0; attempt < 100 && !accepted.ok(); ++attempt) {
    accepted = rund::net::accept::one(rund::node::test::net::ticket(
        listener.view(), rund::net::ready::Interest::Readable));
    if (!accepted.ok()) {
      TEST_ASSERT(accepted.code() == rund::ReasonCode::IoWouldBlock);
      ::usleep(1000);
    }
  }
  TEST_ASSERT(accepted.ok());
  rund::net::Socket accepted_socket = std::move(accepted.socket);

  rund::net::connect::Result finished{};
  for (int attempt = 0; attempt < 100 && !finished.ok(); ++attempt) {
    finished = rund::net::connect::finish(
        rund::node::test::net::ticket(client.view(),
                                      rund::net::ready::Interest::Writable),
        connect_address);
    if (!finished.ok()) {
      ::usleep(1000);
    }
  }
  TEST_ASSERT(finished.ok());

  std::array<std::byte, 2> out{std::byte{'a'}, std::byte{'c'}};
  std::array<std::byte, 2> in{};
  TEST_ASSERT(
      rund::net::send(rund::node::test::net::ticket(
                          client.view(), rund::net::ready::Interest::Writable),
                      std::span<const std::byte>{out})
          .ok());
  TEST_ASSERT(rund::net::nonblocking(accepted_socket.view(), true).ok());
  rund::net::ReceiveResult received{};
  for (int attempt = 0; attempt < 100 && !received.ok(); ++attempt) {
    received = rund::net::receive(
        rund::node::test::net::ticket(accepted_socket.view(),
                                      rund::net::ready::Interest::Readable),
        std::span<std::byte>{in});
    if (!received.ok()) {
      TEST_ASSERT(received.code() == rund::ReasonCode::IoWouldBlock);
      ::usleep(1000);
    }
  }
  TEST_ASSERT(received.ok());
  TEST_ASSERT(in == out);

  return 0;
}
