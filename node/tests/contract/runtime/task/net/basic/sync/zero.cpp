#include "local.hpp"

#include "test/assert.hpp"

#include <array>
#include <cstddef>
#include <rund/net/bytes.hpp>
#include <span>

int RunNetBasicSyncZeroCase() {
  BasicSyncSockets sockets{};
  TEST_ASSERT(OpenBasicSyncSockets(sockets));

  std::array<std::byte, 1u> zero_out{std::byte{'z'}};
  std::array<std::byte, 1u> zero_in{};
  const rund::net::SendResult zero_sent = rund::net::direct::send(
      sockets.left.view(), std::span<const std::byte>{zero_out.data(), 0u});
  TEST_ASSERT(zero_sent.ok());
  TEST_ASSERT(zero_sent.code() == rund::ReasonCode::Ok);
  TEST_ASSERT(zero_sent.bytes == 0);

  const rund::net::ReceiveResult zero_received = rund::net::direct::receive(
      sockets.right.view(), std::span<std::byte>{zero_in.data(), 0u});
  TEST_ASSERT(zero_received.ok());
  TEST_ASSERT(zero_received.code() == rund::ReasonCode::Ok);
  TEST_ASSERT(zero_received.bytes == 0);
  return 0;
}
