#include "src/host/net/test/ticket.hpp"
#include "local.hpp"
#include <rund/net/bytes.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/vectored.hpp>
#include <rund/task/api.hpp>

#include <array>
#include <cstddef>
#include <span>

bool NetVectoredPartialCompletionHashesCompletedPrefix() {
  SocketPair pair{};
  VECTORED_ASSERT(OpenSocketPair(pair));

  const std::array<std::byte, 4u> payload{std::byte{'a'}, std::byte{'b'},
                                          std::byte{'c'}, std::byte{'d'}};
  const std::array<rund::net::batch::Slice, 1u> send_slices{
      rund::net::batch::Slice{.data = payload.data(), .size = payload.size()},
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
            rund::task::spawn("net-vectored-partial", [&] {
              sent = rund::net::batch::send(
                  rund::node::test::net::ticket(
                      pair.left.view(), rund::net::ready::Interest::Writable),
                  send_slices);
              received = rund::net::batch::receive(
                  rund::node::test::net::ticket(
                      pair.right.view(), rund::net::ready::Interest::Readable),
                  recv_slices);
            });
        joined = rund::task::join(task);
      });

  VECTORED_ASSERT(report.ok());
  VECTORED_ASSERT(joined.ok());
  VECTORED_ASSERT(sent.ok());
  VECTORED_ASSERT(sent.bytes == 4);
  VECTORED_ASSERT(received.ok());
  VECTORED_ASSERT(received.bytes == 4);
  const std::array<std::byte, 2u> expected_first{std::byte{'a'},
                                                 std::byte{'b'}};
  const std::array<std::byte, 2u> expected_second{std::byte{'c'},
                                                  std::byte{'d'}};
  const std::array<std::byte, 2u> expected_third{};
  VECTORED_ASSERT(first == expected_first);
  VECTORED_ASSERT(second == expected_second);
  VECTORED_ASSERT(third == expected_third);
  VECTORED_ASSERT(report.events().size() == 2u);
  VECTORED_ASSERT(report.events()[1].kind ==
                  rund::host::EventKind::NetRecvVectored);
  VECTORED_ASSERT(report.events()[1].completed_bytes == payload.size());
  VECTORED_ASSERT(report.events()[1].payload_hash.value ==
                  rund::host::hash_bytes(payload.data(), payload.size()).value);
  return true;
}
