#include "src/host/net/test/ticket.hpp"
#include "local.hpp"
#include <rund/net/bytes.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/vectored.hpp>
#include <rund/task/api.hpp>

#include <array>
#include <cstddef>
#include <span>
#include <utility>

bool NetVectoredRejectsEmptySlices() {
  struct State {
    SocketPair pair{};
    rund::net::SendResult empty_send{};
    rund::net::ReceiveResult empty_recv{};
    std::array<rund::net::batch::Slice, 1u> zero_send{
        rund::net::batch::Slice{.data = nullptr, .size = 0u},
    };
    std::array<rund::net::batch::Buffer, 1u> zero_recv{
        rund::net::batch::Buffer{.data = nullptr, .size = 0u},
    };
    rund::net::SendResult zero_send_result{};
    rund::net::ReceiveResult zero_recv_result{};
    bool empty_send_consumed = false;
    bool empty_recv_consumed = false;
    bool zero_send_consumed = false;
    bool zero_recv_consumed = false;
    rund::task::Status joined{};
  } state{};
  VECTORED_ASSERT(OpenSocketPair(state.pair));

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
      [&state] {
        const rund::task::Handle task =
            rund::task::spawn("net-vectored-empty", [&state] {
              auto empty_send_ticket = rund::node::test::net::ticket(
                  state.pair.left.view(), rund::net::ready::Interest::Writable);
              state.empty_send = rund::net::batch::send(
                  std::move(empty_send_ticket),
                  std::span<const rund::net::batch::Slice>{});
              state.empty_send_consumed = empty_send_ticket.consumed();

              auto empty_recv_ticket = rund::node::test::net::ticket(
                  state.pair.right.view(),
                  rund::net::ready::Interest::Readable);
              state.empty_recv = rund::net::batch::receive(
                  std::move(empty_recv_ticket),
                  std::span<const rund::net::batch::Buffer>{});
              state.empty_recv_consumed = empty_recv_ticket.consumed();

              auto zero_send_ticket = rund::node::test::net::ticket(
                  state.pair.left.view(), rund::net::ready::Interest::Writable);
              state.zero_send_result = rund::net::batch::send(
                  std::move(zero_send_ticket), state.zero_send);
              state.zero_send_consumed = zero_send_ticket.consumed();

              auto zero_recv_ticket = rund::node::test::net::ticket(
                  state.pair.right.view(),
                  rund::net::ready::Interest::Readable);
              state.zero_recv_result = rund::net::batch::receive(
                  std::move(zero_recv_ticket), state.zero_recv);
              state.zero_recv_consumed = zero_recv_ticket.consumed();
            });
        state.joined = rund::task::join(task);
      });

  VECTORED_ASSERT(report.ok());
  VECTORED_ASSERT(state.joined.ok());
  VECTORED_ASSERT(!state.empty_send.ok());
  VECTORED_ASSERT(state.empty_send.code() == rund::ReasonCode::TaskInvalid);
  VECTORED_ASSERT(!state.empty_recv.ok());
  VECTORED_ASSERT(state.empty_recv.code() == rund::ReasonCode::TaskInvalid);
  VECTORED_ASSERT(state.zero_send_result.ok());
  VECTORED_ASSERT(state.zero_send_result.bytes == 0);
  VECTORED_ASSERT(state.zero_recv_result.ok());
  VECTORED_ASSERT(state.zero_recv_result.bytes == 0);
  VECTORED_ASSERT(state.empty_send_consumed);
  VECTORED_ASSERT(state.empty_recv_consumed);
  VECTORED_ASSERT(state.zero_send_consumed);
  VECTORED_ASSERT(state.zero_recv_consumed);
  VECTORED_ASSERT(report.events().size() == 2u);
  VECTORED_ASSERT(report.events()[0u].requested_bytes == 0u);
  VECTORED_ASSERT(report.events()[0u].completed_bytes == 0u);
  VECTORED_ASSERT(report.events()[1u].requested_bytes == 0u);
  VECTORED_ASSERT(report.events()[1u].completed_bytes == 0u);
  return true;
}
