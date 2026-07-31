#include "test/assert.hpp"

#include "local.hpp"

#include <node/runtime/replay.hpp>

#include <cstddef>
#include <vector>

int RunReplayReactorEvidenceContract() {
  const std::vector<rund::host::Event> events{
      rund::host::Event{
          .sequence = 1u,
          .kind = rund::host::EventKind::IoReady,
          .status = rund::host::Status::Ok,
          .task_id = 7u,
          .host_handle_id = 101u,
          .native_errno = 0,
      },
      rund::host::Event{
          .sequence = 2u,
          .kind = rund::host::EventKind::IoReady,
          .status = rund::host::Status::SyscallFailed,
          .task_id = 8u,
          .host_handle_id = 102u,
          .native_errno = 1,
      },
  };
  const std::vector<std::byte> encoded =
      rund::node::EncodeHostReplayEvents(events);
  std::vector<rund::host::Event> decoded{};
  TEST_ASSERT(rund::node::DecodeHostReplayEvents(encoded, decoded));
  TEST_ASSERT(decoded.size() == 2u);
  TEST_ASSERT(decoded[0].sequence == 1u);
  TEST_ASSERT(decoded[1].sequence == 2u);
  TEST_ASSERT(decoded[0].task_id == 7u);
  TEST_ASSERT(decoded[1].task_id == 8u);
  TEST_ASSERT(decoded[1].status == rund::host::Status::SyscallFailed);
  TEST_ASSERT(decoded[1].native_errno == 1);
  return 0;
}
