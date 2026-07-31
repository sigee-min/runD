#include "src/host/net/test/ticket.hpp"
#include "local.hpp"
#include <rund/net/bytes.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/vectored.hpp>
#include <rund/task/api.hpp>

#include <array>
#include <cstddef>
#include <utility>

bool NetVectoredRejectsNullSlices() {
  SocketPair pair{};
  VECTORED_ASSERT(OpenSocketPair(pair));

  const std::array<rund::net::batch::Slice, 1u> null_send{
      rund::net::batch::Slice{.data = nullptr, .size = 1u},
  };
  const std::array<rund::net::batch::Buffer, 1u> null_recv{
      rund::net::batch::Buffer{.data = nullptr, .size = 1u},
  };
  rund::net::SendResult rejected_null_send{};
  rund::net::ReceiveResult rejected_null_recv{};
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
            rund::task::spawn("net-vectored-null", [&] {
              auto send_ticket = rund::node::test::net::ticket(
                  pair.left.view(), rund::net::ready::Interest::Writable);
              rejected_null_send =
                  rund::net::batch::send(std::move(send_ticket), null_send);
              send_consumed = send_ticket.consumed();

              auto recv_ticket = rund::node::test::net::ticket(
                  pair.right.view(), rund::net::ready::Interest::Readable);
              rejected_null_recv =
                  rund::net::batch::receive(std::move(recv_ticket), null_recv);
              recv_consumed = recv_ticket.consumed();
            });
        joined = rund::task::join(task);
      });

  VECTORED_ASSERT(report.ok());
  VECTORED_ASSERT(joined.ok());
  VECTORED_ASSERT(!rejected_null_send.ok());
  VECTORED_ASSERT(rejected_null_send.code() == rund::ReasonCode::TaskInvalid);
  VECTORED_ASSERT(!rejected_null_recv.ok());
  VECTORED_ASSERT(rejected_null_recv.code() == rund::ReasonCode::TaskInvalid);
  VECTORED_ASSERT(send_consumed);
  VECTORED_ASSERT(recv_consumed);
  VECTORED_ASSERT(report.events().empty());
  return true;
}
