#include "src/host/net/test/socket.hpp"
#include "local.hpp"

#include "test/assert.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <rund/net/bytes.hpp>
#include <span>

int RunNetBasicSyncTransferCase() {
  BasicSyncSockets sockets{};
  TEST_ASSERT(OpenBasicSyncSockets(sockets));
  TEST_ASSERT(rund::node::test::net::native(sockets.left) >= 0);
  TEST_ASSERT(rund::node::test::net::native(sockets.right) >= 0);
  TEST_ASSERT(
      sockets.left.id() ==
      static_cast<std::uint64_t>(rund::node::test::net::native(sockets.left)) +
          1u);
  TEST_ASSERT(
      sockets.right.id() ==
      static_cast<std::uint64_t>(rund::node::test::net::native(sockets.right)) +
          1u);
  TEST_ASSERT(rund::node::test::net::admit(-1).id() == 0u);

  std::array<std::byte, 4u> out{std::byte{'p'}, std::byte{'i'}, std::byte{'n'},
                                std::byte{'g'}};
  const rund::net::SendResult sent = rund::net::direct::send(
      sockets.left.view(), std::span<const std::byte>{out});
  TEST_ASSERT(sent.ok());
  TEST_ASSERT(sent.bytes == static_cast<std::int64_t>(out.size()));

  std::array<std::byte, 4u> in{};
  const rund::net::ReceiveResult received = rund::net::direct::receive(
      sockets.right.view(), std::span<std::byte>{in});
  TEST_ASSERT(received.ok());
  TEST_ASSERT(received.bytes == static_cast<std::int64_t>(in.size()));
  TEST_ASSERT(in == out);
  return 0;
}
