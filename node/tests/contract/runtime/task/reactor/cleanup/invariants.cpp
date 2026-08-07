#include "src/host/net/test/socket.hpp"
#include "test/assert.hpp"

#include <rund/net/cancel.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/ready/timed.hpp>
#include <rund/net/socket.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>
#include <rund/task/cancel.hpp>

#include "../../../../../../src/runtime/task/scheduler/reactor/invariants.hpp"
#include "../../../../../../src/runtime/task/scheduler/reactor/many/park/local.hpp"
#include "../../coroutine/allocation.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <string_view>

#include <sys/socket.h>
#include <unistd.h>

namespace {

struct SocketPairCleanup {
  int left = -1;
  int right = -1;

  ~SocketPairCleanup() {
    if (left >= 0) {
      static_cast<void>(::close(left));
    }
    if (right >= 0) {
      static_cast<void>(::close(right));
    }
  }

  SocketPairCleanup() = default;
  SocketPairCleanup(const SocketPairCleanup &) = delete;
  SocketPairCleanup &operator=(const SocketPairCleanup &) = delete;
};

[[nodiscard]] bool MakeSocketPair(SocketPairCleanup &cleanup) noexcept {
  int fds[2] = {-1, -1};
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
    return false;
  }
  cleanup.left = fds[0];
  cleanup.right = fds[1];
  return true;
}

[[nodiscard]] rund::SessionConfig ReactorCleanupInvariantRunSpec() noexcept {
  return rund::SessionConfig{
      .workers = 1u,
      .scheduler =
          {
              .task_capacity = 4u,
              .ready_queue_capacity = 4u,
              .timer_capacity = 4u,
              .reactor_wait_capacity = 4u,
              .observation_capacity = 32u,
              .host_event_capacity = 32u,
          },
  };
}

enum class ReadyManyPublicationFailure : std::uint8_t {
  EventSlots,
  Group,
};

int CheckReadyManyRollbackRequest() {
  constexpr std::uint64_t group_id = 37u;
  constexpr rund::ReasonCode reason =
      rund::ReasonCode::TimerCapacityExceeded;
  const rund::node::ReactorCleanupRequest request =
      rund::node::ReadyManyParkRollbackRequest(group_id, reason);
  TEST_ASSERT(request.wait_id == 0u);
  TEST_ASSERT(request.group_id == group_id);
  TEST_ASSERT(request.reason == reason);
  TEST_ASSERT(!request.cancel_timeout_timer);
  TEST_ASSERT(!request.require_timeout_timer_cancel);
  TEST_ASSERT(request.remove_ready_backlog);
  TEST_ASSERT(request.cleanup_siblings);
  TEST_ASSERT(request.erase_group);
  TEST_ASSERT(!request.wake_owner);
  TEST_ASSERT(request.events == rund::node::ReactorEvent::None);
  TEST_ASSERT(!request.store_event);
  TEST_ASSERT(request.deadline_ns == 0);
  return 0;
}

int CheckReadyManyPublicationRollback(
    const ReadyManyPublicationFailure failure) {
  rund::node::SchedulerState state{};
  rund::node::TaskRecord record{.id = 17u};
  std::array<rund::node::ReactorManyRequest, 2u> requests{};
  rund::node::ReadyManyEntry entry{
      .record = &record,
      .output_limit = static_cast<std::uint32_t>(requests.size()),
      .requests = requests,
      .code = rund::ReasonCode::Ok,
  };
  state.reactor.reactor_many_requests.reserve(requests.size());
  if (failure == ReadyManyPublicationFailure::Group) {
    state.reactor.reactor_many_event_slots.reserve(requests.size());
  }

  constexpr std::uint64_t failed_group_id = 41u;
  const std::uint64_t first_wait_id = state.identity.next_wait_id;
  runtime_task_allocation::FailNext();
  TEST_ASSERT(!rund::node::ReadyManyParkPublishGroup(
      state, entry, failed_group_id, 0u, {}));
  TEST_ASSERT(state.reactor.reactor_many_requests.empty());
  TEST_ASSERT(state.reactor.reactor_many_event_slots.empty());
  TEST_ASSERT(state.reactor.reactor_many_groups.empty());
  TEST_ASSERT(state.identity.next_wait_id == first_wait_id + requests.size());

  constexpr std::uint64_t retry_group_id = failed_group_id + 1u;
  const std::uint64_t retry_first_wait_id = state.identity.next_wait_id;
  TEST_ASSERT(rund::node::ReadyManyParkPublishGroup(
      state, entry, retry_group_id, 0u, {}));
  TEST_ASSERT(state.reactor.reactor_many_requests.size() == requests.size());
  TEST_ASSERT(state.reactor.reactor_many_event_slots.size() == requests.size());
  TEST_ASSERT(state.reactor.reactor_many_groups.size() == 1u);
  TEST_ASSERT(state.reactor.reactor_many_groups.front().group_id ==
              retry_group_id);
  TEST_ASSERT(state.reactor.reactor_many_requests.front().wait_id ==
              retry_first_wait_id);
  TEST_ASSERT(state.identity.next_wait_id ==
              retry_first_wait_id + requests.size());
  return 0;
}

} // namespace

int RunRuntimeTaskReactorCleanupInvariantsContract() {
  TEST_ASSERT(CheckReadyManyRollbackRequest() == 0);
  TEST_ASSERT(CheckReadyManyPublicationRollback(
                  ReadyManyPublicationFailure::EventSlots) == 0);
  TEST_ASSERT(CheckReadyManyPublicationRollback(
                  ReadyManyPublicationFailure::Group) == 0);

  const rund::node::ReactorInvariantSnapshot no_scheduler =
      rund::node::ValidateReactorCleanupInvariantsForTest();
  TEST_ASSERT(!no_scheduler.ok);
  TEST_ASSERT(no_scheduler.reason == std::string_view{"no_scheduler"});

  SocketPairCleanup cleanup{};
  TEST_ASSERT(MakeSocketPair(cleanup));
  rund::net::Socket reader = rund::node::test::net::admit(cleanup.left);
  cleanup.left = -1;
  TEST_ASSERT(rund::net::nonblocking(reader.view(), true).ok());

  rund::node::ReactorInvariantSnapshot live_snapshot{};
  rund::node::ReactorInvariantSnapshot cleanup_snapshot{};
  rund::task::Status joined{};
  rund::net::ready::Ticket ready{};
  bool source_valid = false;
  bool token_valid = false;
  bool cancel_ok = false;

  const rund::Session::Result report =
      rund::run(ReactorCleanupInvariantRunSpec(), [&] {
        auto source = rund::task::stop_source::create();
        source_valid = static_cast<bool>(source);
        if (!source_valid) {
          return;
        }
        auto token = source.token();
        token_valid = static_cast<bool>(token);
        if (!token_valid) {
          return;
        }

        auto wait = [&]() -> rund::task::Task<void> {
          ready = co_await rund::net::ready::timed::read(
              reader.view(), std::chrono::seconds{30}, token);
        };
        const rund::task::Handle waiter =
            rund::task::spawn("reactor-cleanup-invariants-waiter", wait());
        auto cancel = [&]() -> rund::task::Task<void> {
          (void)co_await rund::task::yield();
          live_snapshot = rund::node::ValidateReactorCleanupInvariantsForTest();
          cancel_ok = source.request_stop().ok();
        };
        const rund::task::Handle canceller =
            rund::task::spawn("reactor-cleanup-invariants-canceller", cancel());
        joined = rund::task::join(waiter, canceller);
        cleanup_snapshot =
            rund::node::ValidateReactorCleanupInvariantsForTest();
      });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(source_valid);
  TEST_ASSERT(token_valid);
  TEST_ASSERT(cancel_ok);
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(!ready.ok());
  TEST_ASSERT(ready.code() == rund::ReasonCode::TaskCancelled);
  TEST_ASSERT(live_snapshot.ok);
  TEST_ASSERT(live_snapshot.reason == std::string_view{"ok"});
  TEST_ASSERT(live_snapshot.waits == 1u);
  TEST_ASSERT(live_snapshot.timeout_timers == 1u);
  TEST_ASSERT(live_snapshot.ready_backlog_entries == 0u);
  TEST_ASSERT(live_snapshot.many_groups == 0u);
  TEST_ASSERT(live_snapshot.ready_set_waits == 0u);
  TEST_ASSERT(cleanup_snapshot.ok);
  TEST_ASSERT(cleanup_snapshot.reason == std::string_view{"ok"});
  TEST_ASSERT(cleanup_snapshot.waits == 0u);
  TEST_ASSERT(cleanup_snapshot.timeout_timers == 0u);
  TEST_ASSERT(cleanup_snapshot.ready_backlog_entries == 0u);
  TEST_ASSERT(cleanup_snapshot.many_groups == 0u);
  TEST_ASSERT(cleanup_snapshot.ready_set_waits == 0u);
  return 0;
}
