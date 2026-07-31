#include "src/host/net/test/ticket.hpp"
#include "local.hpp"
#include <rund/net/bytes.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/vectored.hpp>
#include <rund/task/api.hpp>

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

bool NetVectoredRejectsCapacityOverflow() {
  SocketPair pair{};
  VECTORED_ASSERT(OpenSocketPair(pair));

  std::array<std::byte, 1u> one_byte{std::byte{'x'}};
  std::vector<rund::net::batch::Slice> too_many_send(
      static_cast<std::size_t>(rund::SchedulerConfig{}.net_iov_capacity) + 1u,
      rund::net::batch::Slice{.data = one_byte.data(), .size = 1u});
  std::vector<rund::net::batch::Buffer> too_many_recv(
      static_cast<std::size_t>(rund::SchedulerConfig{}.net_iov_capacity) + 1u,
      rund::net::batch::Buffer{.data = one_byte.data(), .size = 1u});

  rund::net::SendResult rejected_many_send{};
  rund::net::ReceiveResult rejected_many_recv{};
  bool send_consumed = false;
  bool recv_consumed = false;
  rund::task::Status joined{};
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_workers = 1u,
                  .task_capacity = 2u,
                  .ready_queue_capacity = 4u,
                  .host_event_capacity = 8u,
              },
      },
      [&] {
        const rund::task::Handle task =
            rund::task::spawn("net-vectored-capacity", [&] {
              auto send_ticket = rund::node::test::net::ticket(
                  pair.left.view(), rund::net::ready::Interest::Writable);
              rejected_many_send =
                  rund::net::batch::send(std::move(send_ticket), too_many_send);
              send_consumed = send_ticket.consumed();

              auto recv_ticket = rund::node::test::net::ticket(
                  pair.right.view(), rund::net::ready::Interest::Readable);
              rejected_many_recv = rund::net::batch::receive(
                  std::move(recv_ticket), too_many_recv);
              recv_consumed = recv_ticket.consumed();
            });
        joined = rund::task::join(task);
      });

  VECTORED_ASSERT(report.ok());
  VECTORED_ASSERT(joined.ok());
  VECTORED_ASSERT(!rejected_many_send.ok());
  VECTORED_ASSERT(rejected_many_send.code() == rund::ReasonCode::TaskInvalid);
  VECTORED_ASSERT(!rejected_many_recv.ok());
  VECTORED_ASSERT(rejected_many_recv.code() == rund::ReasonCode::TaskInvalid);
  VECTORED_ASSERT(send_consumed);
  VECTORED_ASSERT(recv_consumed);
  VECTORED_ASSERT(report.events().empty());
  return true;
}
