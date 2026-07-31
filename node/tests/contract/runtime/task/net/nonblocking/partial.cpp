#include "src/host/net/test/ticket.hpp"
#include "src/host/net/test/socket.hpp"
#include "local.hpp"

#include "test/assert.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <rund/net/bytes.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/socket.hpp>
#include <span>
#include <utility>

int RunNetNonblockingPartialAndNullCase() {
  int sockets[2] = {-1, -1};
  TEST_ASSERT(MakeSocketPair(sockets));
  NonblockingSocketCleanup left_cleanup{sockets[0]};
  NonblockingSocketCleanup right_cleanup{sockets[1]};
  rund::net::Socket left = rund::node::test::net::admit(left_cleanup.fd);
  rund::net::Socket right = rund::node::test::net::admit(right_cleanup.fd);
  TEST_ASSERT(rund::net::nonblocking(left.view(), true).ok());
  TEST_ASSERT(rund::net::nonblocking(right.view(), true).ok());

  std::array<std::byte, 4> partial_out{std::byte{'p'}, std::byte{'a'},
                                       std::byte{'r'}, std::byte{'t'}};
  std::array<std::byte, 2> partial_in{};
  const rund::net::SendResult partial_seed =
      rund::net::send(rund::node::test::net::ticket(
                          left.view(), rund::net::ready::Interest::Writable),
                      std::span<const std::byte>{partial_out});
  TEST_ASSERT(partial_seed.ok());
  const rund::net::ReceiveResult partial_recv = rund::net::receive(
      rund::node::test::net::ticket(right.view(),
                                    rund::net::ready::Interest::Readable),
      std::span<std::byte>{partial_in});
  TEST_ASSERT(partial_recv.ok());
  TEST_ASSERT(partial_recv.bytes ==
              static_cast<std::int64_t>(partial_in.size()));
  TEST_ASSERT(partial_in[0] == partial_out[0]);
  TEST_ASSERT(partial_in[1] == partial_out[1]);

  rund::net::ready::Ticket null_receive = rund::node::test::net::ticket(
      right.view(), rund::net::ready::Interest::Readable);
  const rund::net::ReceiveResult null_try_recv = rund::net::receive(
      std::move(null_receive),
      std::span<std::byte>{static_cast<std::byte *>(nullptr), 1u});
  TEST_ASSERT(!null_try_recv.ok());
  TEST_ASSERT(null_try_recv.code() == rund::ReasonCode::TaskInvalid);
  TEST_ASSERT(null_receive.consumed());
  const rund::net::ReceiveResult reused_receive =
      rund::net::receive(std::move(null_receive), partial_in);
  TEST_ASSERT(!reused_receive);
  TEST_ASSERT(reused_receive.code() == rund::ReasonCode::NetTicketConsumed);
  rund::net::ready::Ticket null_send = rund::node::test::net::ticket(
      left.view(), rund::net::ready::Interest::Writable);
  const rund::net::SendResult null_try_send = rund::net::send(
      std::move(null_send),
      std::span<const std::byte>{static_cast<const std::byte *>(nullptr), 1u});
  TEST_ASSERT(!null_try_send.ok());
  TEST_ASSERT(null_try_send.code() == rund::ReasonCode::TaskInvalid);
  TEST_ASSERT(null_send.consumed());
  const rund::net::SendResult reused_send =
      rund::net::send(std::move(null_send), partial_out);
  TEST_ASSERT(!reused_send);
  TEST_ASSERT(reused_send.code() == rund::ReasonCode::NetTicketConsumed);
  return 0;
}
