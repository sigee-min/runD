#include "src/host/net/test/ticket.hpp"
#include "src/host/net/test/socket.hpp"
#include "../local.hpp"
#include <rund/net/bytes.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/vectored.hpp>
#include <rund/task/api.hpp>

#include <array>
#include <cstddef>

#include <fcntl.h>

bool NetVectoredReportsWouldBlock() {
  SocketPair pair{};
  VECTORED_ASSERT(OpenSocketPair(pair));
  const int flags_before =
      ::fcntl(rund::node::test::net::native(pair.right), F_GETFL, 0);
  VECTORED_ASSERT(flags_before >= 0);
  VECTORED_ASSERT((flags_before & O_NONBLOCK) == 0);

  std::array<std::byte, 1u> one_byte{std::byte{'x'}};
  rund::net::ReceiveResult would_block{};
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
            rund::task::spawn("net-vectored-would-block", [&] {
              const std::array<rund::net::batch::Buffer, 1u> valid_recv{
                  rund::net::batch::Buffer{.data = one_byte.data(),
                                           .size = 1u}};
              would_block = rund::net::batch::receive(
                  rund::node::test::net::ticket(
                      pair.right.view(), rund::net::ready::Interest::Readable),
                  valid_recv);
            });
        joined = rund::task::join(task);
      });

  VECTORED_ASSERT(report.ok());
  VECTORED_ASSERT(joined.ok());
  VECTORED_ASSERT(!would_block.ok());
  VECTORED_ASSERT(would_block.code() == rund::ReasonCode::IoWouldBlock);
  VECTORED_ASSERT(report.events().size() == 1u);
  VECTORED_ASSERT(report.events()[0].kind ==
                  rund::host::EventKind::NetRecvVectored);
  VECTORED_ASSERT(report.events()[0].status == rund::host::Status::WouldBlock);
  VECTORED_ASSERT(::fcntl(rund::node::test::net::native(pair.right), F_GETFL,
                          0) == flags_before);
  VECTORED_ASSERT(report.tasks().network().would_block() == 1u);
  VECTORED_ASSERT(report.tasks().network().bytes_received() == 0u);
  VECTORED_ASSERT(report.tasks().network().bytes_sent() == 0u);
  return true;
}
