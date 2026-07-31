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
#include <span>

#include <fcntl.h>
#include <sys/socket.h>

int RunNetTryPerCallNonblockingCase() {
  int blocking_pair[2] = {-1, -1};
  TEST_ASSERT(MakeSocketPair(blocking_pair));
  NonblockingSocketCleanup left_cleanup{blocking_pair[0]};
  rund::net::Socket blocking_right =
      rund::node::test::net::admit(blocking_pair[1]);
  TEST_ASSERT(blocking_right);
  const int admitted_right = rund::node::test::net::native(blocking_right);
  TEST_ASSERT(admitted_right >= 0);
#if defined(SO_NOSIGPIPE)
  int no_sigpipe = 0;
  socklen_t no_sigpipe_size = sizeof(no_sigpipe);
  TEST_ASSERT(::getsockopt(admitted_right, SOL_SOCKET, SO_NOSIGPIPE,
                           &no_sigpipe, &no_sigpipe_size) == 0);
  TEST_ASSERT(no_sigpipe == 1);
#endif
  const int flags_before = ::fcntl(admitted_right, F_GETFL, 0);
  TEST_ASSERT(flags_before >= 0);
  TEST_ASSERT((flags_before & O_NONBLOCK) == 0);
  std::array<std::byte, 4> in{};
  rund::net::ReceiveResult blocking_task_try{};
  rund::task::Status join_result{};
  const rund::Session::Result blocking_report =
      rund::run(NetNonblockingRunSpec(2u), [&] {
        const rund::task::Handle task =
            rund::task::spawn("net-try-per-call-nonblocking", [&] {
              blocking_task_try =
                  rund::net::receive(rund::node::test::net::ticket(
                                         blocking_right.view(),
                                         rund::net::ready::Interest::Readable),
                                     std::span<std::byte>{in});
            });
        join_result = rund::task::join(task);
      });
  TEST_ASSERT(blocking_report.ok());
  TEST_ASSERT(join_result.ok());
  TEST_ASSERT(!blocking_task_try.ok());
  TEST_ASSERT(blocking_task_try.code() == rund::ReasonCode::IoWouldBlock);
  const int flags_after = ::fcntl(admitted_right, F_GETFL, 0);
  TEST_ASSERT(flags_after == flags_before);
  TEST_ASSERT(blocking_report.events().size() == 1u);
  TEST_ASSERT(blocking_report.events()[0].status ==
              rund::host::Status::WouldBlock);
  return 0;
}
