#include "src/host/net/test/socket.hpp"
#include "local.hpp"
#include <rund/net/bytes.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>

#include "test/assert.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <string_view>

#include <sys/socket.h>

int RunNetBasicTaskGuardCase() {
  int sockets[2] = {-1, -1};
  TEST_ASSERT(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
  BasicSocketCleanup left_cleanup{sockets[0]};
  BasicSocketCleanup right_cleanup{sockets[1]};
  rund::net::Socket event_left = rund::node::test::net::admit(left_cleanup.fd);
  rund::net::Socket event_right =
      rund::node::test::net::admit(right_cleanup.fd);
  std::array<std::byte, 2> event_out{std::byte{'n'}, std::byte{'b'}};
  std::array<std::byte, 2> event_in{};
  rund::net::SendResult task_send{};
  rund::net::ReceiveResult task_recv{};
  rund::task::Status task_join{};
  const rund::Session::Result task_report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 2u,
                  .ready_queue_capacity = 2u,
              },
      },
      [&] {
        const rund::task::Handle task =
            rund::task::spawn("net-direct-guard", [&] {
              task_send = rund::net::direct::send(
                  event_left.view(), std::span<const std::byte>{event_out});
              task_recv = rund::net::direct::receive(
                  event_right.view(), std::span<std::byte>{event_in});
            });
        task_join = rund::task::join(task);
      });
  TEST_ASSERT(task_report.ok());
  TEST_ASSERT(task_join.ok());
  TEST_ASSERT(!task_send.ok());
  TEST_ASSERT(task_send.code() == rund::ReasonCode::TaskInvalid);
  TEST_ASSERT(std::string_view{task_send.error()} == "task_invalid");
  TEST_ASSERT(!task_recv.ok());
  TEST_ASSERT(task_recv.code() == rund::ReasonCode::TaskInvalid);
  TEST_ASSERT(std::string_view{task_recv.error()} == "task_invalid");
  return 0;
}
