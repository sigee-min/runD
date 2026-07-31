#include "src/host/net/test/ticket.hpp"
#include "src/host/net/test/socket.hpp"
#include "local.hpp"

#include "test/assert.hpp"

#include <cstddef>
#include <rund/net/bytes.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/socket.hpp>
#include <span>
#include <vector>

#include <sys/socket.h>

int RunNetNonblockingPressureCase() {
  int pressure_pair[2] = {-1, -1};
  TEST_ASSERT(MakeSocketPair(pressure_pair));
  NonblockingSocketCleanup left_cleanup{pressure_pair[0]};
  NonblockingSocketCleanup right_cleanup{pressure_pair[1]};
  rund::net::Socket pressure_left =
      rund::node::test::net::admit(left_cleanup.fd);
  rund::net::Socket pressure_right =
      rund::node::test::net::admit(right_cleanup.fd);
  TEST_ASSERT(rund::net::nonblocking(pressure_left.view(), true).ok());
  TEST_ASSERT(rund::net::nonblocking(pressure_right.view(), true).ok());
  int small_buffer = 4096;
  static_cast<void>(::setsockopt(rund::node::test::net::native(pressure_left),
                                 SOL_SOCKET, SO_SNDBUF, &small_buffer,
                                 sizeof(small_buffer)));
  std::vector<std::byte> large_payload(1024u * 1024u, std::byte{'s'});
  const rund::net::SendResult partial_send = rund::net::send(
      rund::node::test::net::ticket(pressure_left.view(),
                                    rund::net::ready::Interest::Writable),
      std::span<const std::byte>{large_payload});
  TEST_ASSERT(partial_send.ok());
  TEST_ASSERT(partial_send.bytes > 0);
  TEST_ASSERT(partial_send.bytes <
              static_cast<std::int64_t>(large_payload.size()));
  rund::net::SendResult blocked_send{};
  for (int attempt = 0; attempt < 1024; ++attempt) {
    blocked_send = rund::net::send(
        rund::node::test::net::ticket(pressure_left.view(),
                                      rund::net::ready::Interest::Writable),
        std::span<const std::byte>{large_payload});
    if (!blocked_send.ok()) {
      break;
    }
  }
  TEST_ASSERT(!blocked_send.ok());
  TEST_ASSERT(blocked_send.code() == rund::ReasonCode::IoWouldBlock);
  TEST_ASSERT(blocked_send.native_error != 0);
  return 0;
}
