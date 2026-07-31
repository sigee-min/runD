#include "local.hpp"
#include "src/host/net/test/socket.hpp"
#include <rund/net/io.hpp>
#include <rund/net/ready.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/ready/timed.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>

#include "test/assert.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <span>
#include <utility>

int RunNetReadinessInvalidCase() {
  const rund::net::ready::Ticket invalid =
      std::move(rund::net::ready::read(rund::net::SocketView{})).wait();
  TEST_ASSERT(!invalid);
  TEST_ASSERT(invalid.code() == rund::ReasonCode::IoFdInvalid);

  int sockets[2] = {-1, -1};
  TEST_ASSERT(MakeReadinessSocketPair(sockets));
  ReadinessSocketCleanup left_cleanup{sockets[0]};
  ReadinessSocketCleanup right_cleanup{sockets[1]};
  rund::net::Socket left = rund::node::test::net::admit(left_cleanup.fd);
  TEST_ASSERT(rund::net::nonblocking(left.view(), true).ok());

  std::array<std::byte, 1u> byte{std::byte{'x'}};
  auto invalid_operation =
      rund::net::receive(rund::net::SocketView{}, std::span<std::byte>{byte});
  auto invalid_awaiter = std::move(invalid_operation).operator co_await();
  TEST_ASSERT(invalid_awaiter.await_ready());
  const rund::net::ReceiveResult invalid_receive =
      invalid_awaiter.await_resume();
  TEST_ASSERT(!invalid_receive);
  TEST_ASSERT(invalid_receive.code() == rund::ReasonCode::IoFdInvalid);

  auto missing_operation =
      rund::net::send(left.view(), std::span<const std::byte>{byte});
  auto missing_awaiter = std::move(missing_operation).operator co_await();
  TEST_ASSERT(missing_awaiter.await_ready());
  const rund::net::SendResult missing_send = missing_awaiter.await_resume();
  TEST_ASSERT(!missing_send);
  TEST_ASSERT(missing_send.code() == rund::ReasonCode::NodeRuntimeMissing);
  auto repeated_awaiter = std::move(missing_operation).operator co_await();
  TEST_ASSERT(repeated_awaiter.await_ready());
  const rund::net::SendResult repeated_send = repeated_awaiter.await_resume();
  TEST_ASSERT(!repeated_send);
  TEST_ASSERT(repeated_send.code() == rund::ReasonCode::IoFdInvalid);

  const rund::net::ready::Ticket missing_runtime =
      std::move(rund::net::ready::write(left.view())).wait();
  TEST_ASSERT(!missing_runtime);
  TEST_ASSERT(missing_runtime.code() == rund::ReasonCode::NodeRuntimeMissing);

  rund::net::ready::Ticket invalid_misuse{};
  rund::net::ready::Ticket valid_misuse{};
  rund::net::ready::Ticket invalid_timed_misuse{};
  rund::net::ready::Ticket valid_timed_misuse{};
  rund::net::ReceiveResult invalid_direct_receive{};
  rund::net::SendResult invalid_direct_send{};
  rund::net::ReceiveResult invalid_direct_shape{};
  rund::task::Status joined{};
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 2u,
                  .ready_queue_capacity = 2u,
              },
      },
      [&] {
        auto misuse = [&]() -> rund::task::Task<void> {
          invalid_direct_receive = co_await rund::net::receive(
              rund::net::SocketView{}, std::span<std::byte>{byte});
          invalid_direct_send = co_await rund::net::send(
              rund::net::SocketView{}, std::span<const std::byte>{byte});
          invalid_direct_shape = co_await rund::net::receive(
              left.view(),
              std::span<std::byte>{static_cast<std::byte *>(nullptr), 1u});
          invalid_misuse =
              std::move(rund::net::ready::read(rund::net::SocketView{})).wait();
          valid_misuse = std::move(rund::net::ready::write(left.view())).wait();
          invalid_timed_misuse =
              std::move(rund::net::ready::timed::read(rund::net::SocketView{},
                                                      std::chrono::seconds{1}))
                  .wait();
          valid_timed_misuse =
              std::move(rund::net::ready::timed::write(left.view(),
                                                       std::chrono::seconds{1}))
                  .wait();
          co_return;
        };
        const rund::task::Handle task =
            rund::task::spawn("net-readiness-wait-misuse", misuse());
        joined = rund::task::join(task);
      });
  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(!invalid_misuse);
  TEST_ASSERT(invalid_misuse.code() == rund::ReasonCode::IoFdInvalid);
  TEST_ASSERT(!valid_misuse);
  TEST_ASSERT(valid_misuse.code() == rund::ReasonCode::TaskContextMissing);
  TEST_ASSERT(!invalid_timed_misuse);
  TEST_ASSERT(invalid_timed_misuse.code() == rund::ReasonCode::IoFdInvalid);
  TEST_ASSERT(!valid_timed_misuse);
  TEST_ASSERT(valid_timed_misuse.code() ==
              rund::ReasonCode::TaskContextMissing);
  TEST_ASSERT(!invalid_direct_receive);
  TEST_ASSERT(invalid_direct_receive.code() == rund::ReasonCode::IoFdInvalid);
  TEST_ASSERT(!invalid_direct_send);
  TEST_ASSERT(invalid_direct_send.code() == rund::ReasonCode::IoFdInvalid);
  TEST_ASSERT(!invalid_direct_shape);
  TEST_ASSERT(invalid_direct_shape.code() == rund::ReasonCode::TaskInvalid);
  TEST_ASSERT(report.tasks().network().send_calls() == 0u);
  TEST_ASSERT(report.tasks().network().recv_calls() == 0u);
  TEST_ASSERT(report.tasks().reactor_waits() == 0u);
  return 0;
}
