#include "local.hpp"

#include "test/assert.hpp"

#include <array>
#include <cstddef>
#include <rund/net/bytes.hpp>
#include <rund/net/socket.hpp>
#include <span>
#include <string_view>

int RunNetBasicSyncInvalidCase() {
  BasicSyncSockets sockets{};
  TEST_ASSERT(OpenBasicSyncSockets(sockets));
  std::array<std::byte, 4u> in{};

  const rund::net::ReceiveResult invalid = rund::net::direct::receive(
      rund::net::SocketView{}, std::span<std::byte>{in});
  TEST_ASSERT(!invalid.ok());
  TEST_ASSERT(std::string_view{invalid.error()} == "io_fd_invalid");

  const rund::net::ReceiveResult null_nonempty_recv =
      rund::net::direct::receive(
          sockets.right.view(),
          std::span<std::byte>{static_cast<std::byte *>(nullptr), 1u});
  TEST_ASSERT(!null_nonempty_recv.ok());
  TEST_ASSERT(null_nonempty_recv.code() == rund::ReasonCode::TaskInvalid);
  TEST_ASSERT(std::string_view{null_nonempty_recv.error()} == "task_invalid");

  const rund::net::SendResult null_nonempty_send = rund::net::direct::send(
      sockets.left.view(),
      std::span<const std::byte>{static_cast<const std::byte *>(nullptr), 1u});
  TEST_ASSERT(!null_nonempty_send.ok());
  TEST_ASSERT(null_nonempty_send.code() == rund::ReasonCode::TaskInvalid);
  TEST_ASSERT(std::string_view{null_nonempty_send.error()} == "task_invalid");

  return 0;
}
