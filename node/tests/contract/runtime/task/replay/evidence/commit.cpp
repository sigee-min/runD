#include "test/assert.hpp"

#include "local.hpp"
#include "src/runtime/task/scheduler/state.hpp"

#include <rund/host/hash.hpp>
#include <rund/replay.hpp>
#include <rund/session.hpp>

#include <string_view>

namespace {

[[nodiscard]] bool CommitNamedEvent(const std::string_view name) noexcept {
  rund::node::Scheduler *const scheduler = rund::node::Scheduler::Active();
  return scheduler != nullptr &&
         scheduler->RecordHostEvent({
             .kind = rund::host::EventKind::EnvGet,
             .status = rund::host::Status::Ok,
             .name_hash = rund::host::hash_string(name.data(), name.size()),
         });
}

} // namespace

int RunReplayHostCommitContract() {
  rund::Session session{};
  rund::SessionConfig config{};
  config.workers = 1u;
  config.scheduler.host_event_capacity = 1u;
  TEST_ASSERT(session.open(config));

  bool recorded_event = false;
  const rund::replay::Record recorded =
      rund::replay::record(session, [&](rund::replay::Context &) {
        recorded_event = CommitNamedEvent("recorded");
      });
  TEST_ASSERT(recorded);
  TEST_ASSERT(recorded_event);
  TEST_ASSERT(recorded.host_event_count() == 1u);

  bool replay_event = true;
  const rund::replay::Check mismatch =
      rund::replay::run(session, recorded, [&](rund::replay::Context &) {
        replay_event = CommitNamedEvent("mismatch");
      });
  TEST_ASSERT(!mismatch);
  TEST_ASSERT(mismatch.code() ==
              rund::replay::Code::HostReplayEventMismatch);
  TEST_ASSERT(!replay_event);
  TEST_ASSERT(mismatch.actual().has_value());
  TEST_ASSERT(mismatch.actual()->host_event_count() == 1u);
  const rund::replay::Window window =
      rund::replay::window(recorded, *mismatch.actual(), 1u);
  TEST_ASSERT(window.host_event_index() == 0u);
  TEST_ASSERT(window.expected_host_events().size() == 1u);
  TEST_ASSERT(window.actual_host_events().size() == 1u);
  TEST_ASSERT(window.expected_host_events()[0].sequence == 1u);
  TEST_ASSERT(window.actual_host_events()[0].sequence == 1u);
  TEST_ASSERT(window.actual_host_events()[0].name_hash.value ==
              rund::host::hash_string("mismatch", 8u).value);
  TEST_ASSERT(mismatch.actual()->tasks().host_events() == 1u);
  TEST_ASSERT(mismatch.actual()->tasks().host_events_dropped() == 0u);

  TEST_ASSERT(session.close());
  return 0;
}
