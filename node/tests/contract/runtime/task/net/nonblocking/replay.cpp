#include "src/host/net/test/ticket.hpp"
#include "src/host/net/test/socket.hpp"
#include "local.hpp"
#include <rund/net/bytes.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>

#include "test/assert.hpp"

#include <rund/replay.hpp>

#include <array>
#include <cstddef>
#include <span>

int RunNetNonblockingReplayCase() {
  int replay_pair[2] = {-1, -1};
  TEST_ASSERT(MakeSocketPair(replay_pair));
  NonblockingSocketCleanup left_cleanup{replay_pair[0]};
  NonblockingSocketCleanup right_cleanup{replay_pair[1]};
  rund::net::Socket replay_right =
      rund::node::test::net::admit(right_cleanup.fd);
  TEST_ASSERT(rund::net::nonblocking(replay_right.view(), true).ok());
  std::array<std::byte, 1> replay_buffer{};
  rund::task::Status join_result{};
  rund::Session session{};
  TEST_ASSERT(session.open(NetNonblockingRunSpec(2u)));
  auto receive = [&](rund::replay::Context &) {
    const rund::task::Handle task =
        rund::task::spawn("net-would-block-record", [&] {
          (void)rund::net::receive(
              rund::node::test::net::ticket(
                  replay_right.view(), rund::net::ready::Interest::Readable),
              std::span<std::byte>{replay_buffer});
        });
    join_result = rund::task::join(task);
  };
  const rund::replay::Record recorded = rund::replay::record(session, receive);
  TEST_ASSERT(recorded.ok());
  TEST_ASSERT(join_result.ok());
  TEST_ASSERT(recorded.host_event_count() == 1u);
  const rund::replay::Check replayed =
      rund::replay::run(session, recorded, receive);
  TEST_ASSERT(replayed.ok());
  TEST_ASSERT(join_result.ok());
  TEST_ASSERT(replayed.actual().has_value());
  TEST_ASSERT(replayed.actual()->host_event_hash() ==
              recorded.host_event_hash());
  TEST_ASSERT(session.close());
  return 0;
}
