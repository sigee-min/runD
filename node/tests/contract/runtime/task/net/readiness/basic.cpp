#include "local.hpp"
#include "src/host/net/test/socket.hpp"
#include <rund/net/io.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>

#include "test/assert.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <utility>

int RunNetReadinessBasicCase() {
  int sockets[2] = {-1, -1};
  TEST_ASSERT(MakeReadinessSocketPair(sockets));
  ReadinessSocketCleanup left_cleanup{sockets[0]};
  ReadinessSocketCleanup right_cleanup{sockets[1]};
  rund::net::Socket left = rund::node::test::net::admit(left_cleanup.fd);
  rund::net::Socket right = rund::node::test::net::admit(right_cleanup.fd);
  TEST_ASSERT(rund::net::nonblocking(left.view(), true).ok());
  TEST_ASSERT(rund::net::nonblocking(right.view(), true).ok());

  std::array<std::byte, 2> out{std::byte{'r'}, std::byte{'d'}};
  std::array<std::byte, 2> in{};
  rund::net::SendResult sent{};
  rund::net::SendResult moved_source{};
  rund::net::SendResult repeated{};
  rund::net::ReceiveResult received{};
  rund::task::Status join_result{};
  auto write_operation =
      rund::net::send(left.view(), std::span<const std::byte>{out});
  auto read_operation =
      rund::net::receive(right.view(), std::span<std::byte>{in});
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 2u,
                  .ready_queue_capacity = 4u,
                  .host_event_capacity = 4u,
              },
      },
      [&, write_operation = std::move(write_operation),
       read_operation = std::move(read_operation)]() mutable {
        auto read = [&, write_operation = std::move(write_operation),
                     read_operation = std::move(
                         read_operation)]() mutable -> rund::task::Task<void> {
          auto moved_write = std::move(write_operation);
          sent = co_await std::move(moved_write);
          moved_source = co_await std::move(write_operation);
          repeated = co_await std::move(moved_write);
          if (sent) {
            received = co_await std::move(read_operation);
          }
        };
        const rund::task::Handle task =
            rund::task::spawn("net-readiness", read());
        join_result = rund::task::join(task);
      });
  TEST_ASSERT(report.ok());
  TEST_ASSERT(join_result.ok());
  TEST_ASSERT(sent);
  TEST_ASSERT(sent.bytes == static_cast<std::int64_t>(out.size()));
  TEST_ASSERT(!moved_source);
  TEST_ASSERT(moved_source.code() == rund::ReasonCode::IoFdInvalid);
  TEST_ASSERT(!repeated);
  TEST_ASSERT(repeated.code() == rund::ReasonCode::IoFdInvalid);
  TEST_ASSERT(received.ok());
  TEST_ASSERT(received.bytes == static_cast<std::int64_t>(in.size()));
  TEST_ASSERT(in == out);
  TEST_ASSERT(report.tasks().spawned() == 1u);
  TEST_ASSERT(report.tasks().coroutine_frame_destroys() == 1u);
  TEST_ASSERT(report.tasks().resources().coroutine_frame_allocations() == 1u);
  TEST_ASSERT(report.tasks().reactor_waits() == 0u);
  TEST_ASSERT(report.tasks().network().send_calls() == 1u);
  TEST_ASSERT(report.tasks().network().recv_calls() == 1u);
  TEST_ASSERT(report.events().size() == 4u);
  TEST_ASSERT(report.events()[0].kind == rund::host::EventKind::IoReady);
  TEST_ASSERT(report.events()[1].kind == rund::host::EventKind::NetSend);
  TEST_ASSERT(report.events()[2].kind == rund::host::EventKind::IoReady);
  TEST_ASSERT(report.events()[3].kind == rund::host::EventKind::NetRecv);
  TEST_ASSERT(report.events()[0].host_handle_id != 0u);
  TEST_ASSERT(report.events()[0].host_handle_id ==
              report.events()[1].host_handle_id);
  TEST_ASSERT(report.events()[2].host_handle_id != 0u);
  TEST_ASSERT(report.events()[2].host_handle_id ==
              report.events()[3].host_handle_id);
  TEST_ASSERT(report.events()[0].host_handle_id !=
              report.events()[2].host_handle_id);
  return 0;
}
