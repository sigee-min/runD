#include "../../../../../src/runtime/reactor/platform.hpp"
#include "../../../../../src/runtime/reactor/diagnostics.hpp"
#include "../../../../../src/runtime/reactor/readiness/handle.hpp"
#include "../../../../../src/runtime/reactor/readiness/mask.hpp"

#include <unistd.h>

#include <vector>

#include "../coroutine/allocation.hpp"
#include "test/assert.hpp"

int RunRuntimeTaskReactorPlatformContract() {
  using namespace rund::node;

  static_assert(sizeof(ReactorHandle) >= sizeof(void*));
  static_assert(ReactorInterestBits(ReactorInterest::Read) == 1);
  static_assert(ReactorInterestBits(ReactorInterest::Write) == 4);

  // This contract is compiled against the neutral interface only. Platform
  // SDK records are deliberately unavailable at this boundary.

  ResetReactorBackendStats();
  ReactorPlatform platform{};

  TEST_ASSERT(PrepareReactorPlatform(platform, 8u).ok);
  ReactorPlatformState* const storage = platform.state.get();
  TEST_ASSERT(storage != nullptr);
  TEST_ASSERT(OpenReactorPlatform(platform).ok);
  CloseReactorPlatform(platform);

  TEST_ASSERT(PrepareReactorPlatform(platform, 16u).ok);
  TEST_ASSERT(platform.state.get() == storage);
  TEST_ASSERT(OpenReactorPlatform(platform).ok);
  CloseReactorPlatform(platform);

  {
    ReactorPlatform scoped{};
    TEST_ASSERT(PrepareReactorPlatform(scoped, 4u).ok);
    TEST_ASSERT(OpenReactorPlatform(scoped).ok);
  }

  int pipe_fds[2] = {-1, -1};
  TEST_ASSERT(::pipe(pipe_fds) == 0);
  const char byte = 'p';
  TEST_ASSERT(::write(pipe_fds[1], &byte, 1u) == 1);
  TEST_ASSERT(OpenReactorPlatform(platform).ok);
  TEST_ASSERT(AddReactorPlatformInterest(platform,
                                         ReactorHandleFromPublic(pipe_fds[0]),
                                         ReactorInterest::Read)
                  .ok);
  const ReactorPlatformPollResult first_poll =
      PollReactorPlatform(platform, 0, 1u);
  TEST_ASSERT(first_poll.ok);
  TEST_ASSERT(first_poll.ready != nullptr);
  TEST_ASSERT(first_poll.ready->size() == 1u);
  TEST_ASSERT(
      HasReactorEvent(first_poll.ready->front().events, ReactorEvent::Read));

  runtime_task_allocation::Start();
  const ReactorPlatformPollResult warm_poll =
      PollReactorPlatform(platform, 0, 1u);
  runtime_task_allocation::Stop();
  TEST_ASSERT(warm_poll.ok);
  TEST_ASSERT(warm_poll.ready != nullptr);
  TEST_ASSERT(warm_poll.ready->size() == 1u);
  TEST_ASSERT(runtime_task_allocation::Count() == 0u);

  char consumed = 0;
  TEST_ASSERT(::read(pipe_fds[0], &consumed, 1u) == 1);
  const ReactorHandle writable = ReactorHandleFromPublic(pipe_fds[1]);
  TEST_ASSERT(
      AddReactorPlatformInterest(platform, writable, ReactorInterest::Write)
          .ok);
  const ReactorPlatformPollResult writable_poll =
      PollReactorPlatform(platform, 0, 1u);
  TEST_ASSERT(writable_poll.ok);
  TEST_ASSERT(writable_poll.ready != nullptr);
  TEST_ASSERT(writable_poll.ready->size() == 1u);
  TEST_ASSERT(writable_poll.ready->front().handle == writable);
  TEST_ASSERT(HasReactorEvent(writable_poll.ready->front().events,
                              ReactorEvent::Write));
  TEST_ASSERT(RemoveReactorPlatformInterest(platform, writable).ok);
  const ReactorPlatformPollResult removed_poll =
      PollReactorPlatform(platform, 0, 1u);
  TEST_ASSERT(removed_poll.ok);
  TEST_ASSERT(removed_poll.ready != nullptr);
  TEST_ASSERT(removed_poll.ready->empty());
  TEST_ASSERT(::write(pipe_fds[1], &byte, 1u) == 1);
  CloseReactorPlatform(platform);

  const BatchIoPollRequest request{
      .index = 0u,
      .handle = ReactorHandleFromPublic(pipe_fds[0]),
      .interest = ReactorInterest::Read,
  };
  std::vector<BatchIoReady> ready{};
  ready.reserve(1u);
  TEST_ASSERT(ProbeReactorPlatformNow(platform, &request, 1u, ready).ok);
  TEST_ASSERT(ready.size() == 1u);

  runtime_task_allocation::Start();
  const BatchIoProbeResult warm =
      ProbeReactorPlatformNow(platform, &request, 1u, ready);
  runtime_task_allocation::Stop();
  TEST_ASSERT(warm.ok);
  TEST_ASSERT(ready.size() == 1u);
  TEST_ASSERT(runtime_task_allocation::Count() == 0u);
  TEST_ASSERT(::close(pipe_fds[0]) == 0);
  TEST_ASSERT(::close(pipe_fds[1]) == 0);

  const ReactorBackendStats stats = ReactorBackendStatsSnapshot();
  TEST_ASSERT(stats.open_calls == 4u);
  TEST_ASSERT(stats.close_calls == 4u);
  TEST_ASSERT(stats.current_open_handles == 0u);
  TEST_ASSERT(stats.max_open_handles == 1u);
  return 0;
}
