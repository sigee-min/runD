#include "src/host/net/test/ticket.hpp"
#include "src/host/net/test/socket.hpp"
#include "local.hpp"
#include <rund/net/bytes.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>

#include "test/assert.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

int RunNetNonblockingBasicCase() {
  int sockets[2] = {-1, -1};
  TEST_ASSERT(MakeSocketPair(sockets));
  NonblockingSocketCleanup left_cleanup{sockets[0]};
  NonblockingSocketCleanup right_cleanup{sockets[1]};
  rund::net::Socket left = rund::node::test::net::admit(left_cleanup.fd);
  rund::net::Socket right = rund::node::test::net::admit(right_cleanup.fd);

  const rund::net::NonblockingResult left_nonblocking =
      rund::net::nonblocking(left.view(), true);
  const rund::net::NonblockingResult right_nonblocking =
      rund::net::nonblocking(right.view(), true);
  TEST_ASSERT(left_nonblocking.ok());
  TEST_ASSERT(right_nonblocking.ok());
  TEST_ASSERT(left_nonblocking.enabled);
  TEST_ASSERT(right_nonblocking.enabled);

  std::array<std::byte, 4> in{};
  std::array<std::byte, 3> out{std::byte{'n'}, std::byte{'b'}, std::byte{'1'}};
  rund::net::ReceiveResult would_block{};
  rund::net::SendResult sent{};
  rund::net::ReceiveResult received{};
  rund::task::Status join_result{};
  const rund::Session::Result report = rund::run(NetNonblockingRunSpec(), [&] {
    const rund::task::Handle task = rund::task::spawn("net-nonblocking", [&] {
      would_block = rund::net::receive(
          rund::node::test::net::ticket(right.view(),
                                        rund::net::ready::Interest::Readable),
          std::span<std::byte>{in});
      sent = rund::net::send(
          rund::node::test::net::ticket(left.view(),
                                        rund::net::ready::Interest::Writable),
          std::span<const std::byte>{out});
      received = rund::net::receive(
          rund::node::test::net::ticket(right.view(),
                                        rund::net::ready::Interest::Readable),
          std::span<std::byte>{in});
    });
    join_result = rund::task::join(task);
  });
  TEST_ASSERT(report.ok());
  TEST_ASSERT(join_result.ok());
  TEST_ASSERT(!would_block.ok());
  TEST_ASSERT(would_block.code() == rund::ReasonCode::IoWouldBlock);
  TEST_ASSERT(std::string_view{would_block.error()} == "io_would_block");
  TEST_ASSERT(sent.ok());
  TEST_ASSERT(sent.bytes == static_cast<std::int64_t>(out.size()));
  TEST_ASSERT(received.ok());
  TEST_ASSERT(received.bytes == static_cast<std::int64_t>(out.size()));
  TEST_ASSERT(in[0] == out[0]);
  TEST_ASSERT(in[1] == out[1]);
  TEST_ASSERT(in[2] == out[2]);
  TEST_ASSERT(report.events().size() == 3u);
  TEST_ASSERT(report.events()[0].kind == rund::host::EventKind::NetRecv);
  TEST_ASSERT(report.events()[0].status == rund::host::Status::WouldBlock);
  TEST_ASSERT(report.events()[0].host_handle_id != 0u);
  TEST_ASSERT(report.events()[0].completed_bytes == 0u);
  TEST_ASSERT(report.events()[1].kind == rund::host::EventKind::NetSend);
  TEST_ASSERT(report.events()[1].status == rund::host::Status::Ok);
  TEST_ASSERT(report.events()[1].host_handle_id != 0u);
  TEST_ASSERT(report.events()[1].host_handle_id !=
              report.events()[0].host_handle_id);
  TEST_ASSERT(report.events()[2].kind == rund::host::EventKind::NetRecv);
  TEST_ASSERT(report.events()[2].status == rund::host::Status::Ok);
  TEST_ASSERT(report.events()[2].host_handle_id ==
              report.events()[0].host_handle_id);
  TEST_ASSERT(report.events()[2].payload_hash.value ==
              rund::host::hash_bytes(in.data(), out.size()).value);
  return 0;
}
