#include "src/host/net/test/ticket.hpp"
#include "local.hpp"
#include <rund/net/bytes.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/vectored.hpp>
#include <rund/task/api.hpp>

#include "../../coroutine/allocation.hpp"

#include <array>
#include <cstddef>
#include <span>

bool NetVectoredSendRecvPreservesSliceOrder() {
  SocketPair pair{};
  VECTORED_ASSERT(OpenSocketPair(pair));

  const std::array<std::byte, 2u> ab{std::byte{'a'}, std::byte{'b'}};
  const std::array<std::byte, 2u> cd{std::byte{'c'}, std::byte{'d'}};
  const std::array<std::byte, 2u> ef{std::byte{'e'}, std::byte{'f'}};
  const std::array<rund::net::batch::Slice, 3u> send_slices{
      rund::net::batch::Slice{.data = ab.data(), .size = ab.size()},
      rund::net::batch::Slice{.data = cd.data(), .size = cd.size()},
      rund::net::batch::Slice{.data = ef.data(), .size = ef.size()},
  };
  std::array<std::byte, 2u> first{};
  std::array<std::byte, 2u> second{};
  std::array<std::byte, 2u> third{};
  const std::array<rund::net::batch::Buffer, 3u> recv_slices{
      rund::net::batch::Buffer{.data = first.data(), .size = first.size()},
      rund::net::batch::Buffer{.data = second.data(), .size = second.size()},
      rund::net::batch::Buffer{.data = third.data(), .size = third.size()},
  };

  rund::net::SendResult sent{};
  rund::net::ReceiveResult received{};
  rund::task::Status joined{};
  std::uint64_t allocations = ~std::uint64_t{0u};
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
            rund::task::spawn("net-vectored-order", [&] {
              runtime_task_allocation::Start();
              sent = rund::net::batch::send(
                  rund::node::test::net::ticket(
                      pair.left.view(), rund::net::ready::Interest::Writable),
                  send_slices);
              received = rund::net::batch::receive(
                  rund::node::test::net::ticket(
                      pair.right.view(), rund::net::ready::Interest::Readable),
                  recv_slices);
              runtime_task_allocation::Stop();
              allocations = runtime_task_allocation::Count();
            });
        joined = rund::task::join(task);
      });

  VECTORED_ASSERT(report.ok());
  VECTORED_ASSERT(joined.ok());
  VECTORED_ASSERT(sent.ok());
  VECTORED_ASSERT(sent.bytes == 6);
  VECTORED_ASSERT(received.ok());
  VECTORED_ASSERT(received.bytes == 6);
  VECTORED_ASSERT(allocations == 0u);
  const std::array<std::byte, 6u> bytes{first[0],  first[1], second[0],
                                        second[1], third[0], third[1]};
  VECTORED_ASSERT(BytesToString(bytes) == "abcdef");
  VECTORED_ASSERT(report.events().size() == 2u);
  VECTORED_ASSERT(report.events()[0].kind ==
                  rund::host::EventKind::NetSendVectored);
  VECTORED_ASSERT(report.events()[1].kind ==
                  rund::host::EventKind::NetRecvVectored);
  for (const rund::host::Event &event : report.events()) {
    VECTORED_ASSERT(event.requested_bytes == bytes.size());
    VECTORED_ASSERT(event.completed_bytes == bytes.size());
    VECTORED_ASSERT(event.payload_hash.value ==
                    rund::host::hash_bytes(bytes.data(), bytes.size()).value);
  }
  VECTORED_ASSERT(report.tasks().network().vectored_send_calls() == 1u);
  VECTORED_ASSERT(report.tasks().network().vectored_recv_calls() == 1u);
  VECTORED_ASSERT(report.tasks().network().bytes_sent() == 6u);
  VECTORED_ASSERT(report.tasks().network().bytes_received() == 6u);
  return true;
}
