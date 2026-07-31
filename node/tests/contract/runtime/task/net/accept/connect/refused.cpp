#include "src/host/net/test/ticket.hpp"
#include "src/host/net/test/socket.hpp"
#include "local.hpp"

#include "test/assert.hpp"

#include <rund/net/address.hpp>
#include <rund/net/connection.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/socket.hpp>
#include <utility>

#include <sys/socket.h>
#include <unistd.h>

int RunAcceptConnectRefusedCase() {
  int closed_listener_fd = -1;
  sockaddr_in closed_listener_address{};
  TEST_ASSERT(MakeAcceptConnectLoopbackListener(&closed_listener_fd,
                                                &closed_listener_address));
  AcceptConnectSocketCleanup closed_listener_cleanup{closed_listener_fd};
  closed_listener_cleanup.reset(-1);
  const int refused_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  TEST_ASSERT(refused_fd >= 0);
  AcceptConnectSocketCleanup refused_cleanup{refused_fd};
  rund::net::Socket refused_socket =
      rund::node::test::net::admit(refused_cleanup.fd);
  TEST_ASSERT(rund::net::nonblocking(refused_socket.view(), true).ok());
  const rund::net::connect::Result invalid_start =
      rund::net::connect::start(refused_socket.view(), rund::net::Address{});
  TEST_ASSERT(invalid_start.code() == rund::ReasonCode::TaskInvalid);

  auto invalid_ticket = rund::node::test::net::ticket(
      refused_socket.view(), rund::net::ready::Interest::Writable);
  const rund::net::connect::Result invalid_finish = rund::net::connect::finish(
      std::move(invalid_ticket), rund::net::Address{});
  TEST_ASSERT(invalid_finish.code() == rund::ReasonCode::TaskInvalid);
  TEST_ASSERT(invalid_ticket.consumed());
  TEST_ASSERT(rund::net::connect::finish(std::move(invalid_ticket),
                                         rund::net::Address{})
                  .code() == rund::ReasonCode::NetTicketConsumed);

  const rund::net::Address refused_address =
      AcceptConnectAddressFromSockaddr(closed_listener_address);
  const rund::net::connect::Result refused_start =
      rund::net::connect::start(refused_socket.view(), refused_address);
  bool finish_refused = false;
  if (refused_start.ok()) {
    for (int attempt = 0; attempt < 200 && !finish_refused; ++attempt) {
      const rund::net::connect::Result refused_finish =
          rund::net::connect::finish(
              rund::node::test::net::ticket(
                  refused_socket.view(), rund::net::ready::Interest::Writable),
              refused_address);
      finish_refused =
          !refused_finish.ok() &&
          refused_finish.code() == rund::ReasonCode::IoSyscallFailed &&
          refused_finish.native_error != 0 &&
          refused_finish.address_hash.value == refused_start.address_hash.value;
      if (!finish_refused) {
        ::usleep(1000);
      }
    }
    TEST_ASSERT(finish_refused);
  } else {
    TEST_ASSERT(refused_start.code() == rund::ReasonCode::IoSyscallFailed);
    TEST_ASSERT(refused_start.native_error != 0);
  }

  return 0;
}
