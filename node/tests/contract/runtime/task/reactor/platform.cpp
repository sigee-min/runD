#include "../../../../../src/runtime/reactor/platform.hpp"
#include "../../../../../src/runtime/reactor/diagnostics.hpp"
#include "../../../../../src/runtime/reactor/readiness/handle.hpp"
#include "../../../../../src/runtime/reactor/readiness/mask.hpp"

#include <array>
#include <cerrno>
#include <sys/socket.h>
#include <type_traits>
#include <unistd.h>

#include <vector>

#include "../coroutine/allocation.hpp"
#include "test/assert.hpp"

namespace {

[[nodiscard]] constexpr bool
OperationSucceeded(const rund::node::ReactorPlatformOpResult result) noexcept {
  return result.disposition() ==
         rund::node::ReactorPlatformOpDisposition::Success;
}

} // namespace

int RunRuntimeTaskReactorPlatformContract() {
  using namespace rund::node;

  static_assert(sizeof(ReactorHandle) >= sizeof(void*));
  static_assert(ReactorInterestBits(ReactorInterest::Read) == 1);
  static_assert(ReactorInterestBits(ReactorInterest::Write) == 4);
  static_assert(!std::is_aggregate_v<ReactorPlatformOpResult>);
  static_assert(std::is_trivially_copyable_v<ReactorPlatformOpResult>);
  static_assert(!std::is_aggregate_v<BatchIoProbeResult>);
  static_assert(std::is_trivially_copyable_v<BatchIoProbeResult>);
  static_assert(!std::is_aggregate_v<ReactorPlatformBatchResult>);
  static_assert(std::is_trivially_copyable_v<ReactorPlatformBatchResult>);
  static_assert(!std::is_aggregate_v<ReactorPlatformPollResult>);
  static_assert(std::is_trivially_copyable_v<ReactorPlatformPollResult>);

  constexpr ReactorPlatformOpResult operation_success =
      ReactorPlatformOpResult::success();
  static_assert(operation_success.disposition() ==
                ReactorPlatformOpDisposition::Success);
  static_assert(operation_success.platform_error() == 0);

  constexpr ReactorPlatformOpResult operation_invalid =
      ReactorPlatformOpResult::invalid(EBADF);
  static_assert(operation_invalid.disposition() ==
                ReactorPlatformOpDisposition::Invalid);
  static_assert(operation_invalid.platform_error() == EBADF);

  constexpr ReactorPlatformOpResult operation_failed =
      ReactorPlatformOpResult::failed(ENOMEM);
  static_assert(operation_failed.disposition() ==
                ReactorPlatformOpDisposition::Failed);
  static_assert(operation_failed.platform_error() == ENOMEM);

  constexpr ReactorPlatformOpResult operation_unavailable =
      ReactorPlatformOpResult::backend_unavailable();
  static_assert(operation_unavailable.disposition() ==
                ReactorPlatformOpDisposition::BackendUnavailable);
  static_assert(operation_unavailable.platform_error() == 0);

  constexpr ReactorPlatformBatchResult batch_success =
      ReactorPlatformBatchResult::success();
  static_assert(batch_success.disposition() ==
                ReactorPlatformBatchDisposition::Success);
  static_assert(batch_success.platform_error() == 0);
  static_assert(batch_success.failed_index() == 0u);

  constexpr ReactorPlatformBatchResult batch_invalid =
      ReactorPlatformBatchResult::invalid(EBADF, 3u);
  static_assert(batch_invalid.disposition() ==
                ReactorPlatformBatchDisposition::Invalid);
  static_assert(batch_invalid.platform_error() == EBADF);
  static_assert(batch_invalid.failed_index() == 3u);

  constexpr ReactorPlatformBatchResult batch_failed =
      ReactorPlatformBatchResult::failed(ENOMEM, 4u);
  static_assert(batch_failed.disposition() ==
                ReactorPlatformBatchDisposition::Failed);
  static_assert(batch_failed.platform_error() == ENOMEM);
  static_assert(batch_failed.failed_index() == 4u);

  constexpr ReactorPlatformBatchResult batch_unavailable =
      ReactorPlatformBatchResult::backend_unavailable();
  static_assert(batch_unavailable.disposition() ==
                ReactorPlatformBatchDisposition::BackendUnavailable);
  static_assert(batch_unavailable.platform_error() == 0);
  static_assert(batch_unavailable.failed_index() == 0u);

  constexpr ReactorPlatformPollResult poll_success =
      ReactorPlatformPollResult::success();
  static_assert(poll_success.disposition() ==
                ReactorPlatformPollDisposition::Success);
  static_assert(poll_success.platform_error() == 0);

  constexpr ReactorPlatformPollResult poll_invalid =
      ReactorPlatformPollResult::invalid(EBADF);
  static_assert(poll_invalid.disposition() ==
                ReactorPlatformPollDisposition::Invalid);
  static_assert(poll_invalid.platform_error() == EBADF);

  constexpr ReactorPlatformPollResult poll_failed =
      ReactorPlatformPollResult::failed(ENOMEM);
  static_assert(poll_failed.disposition() ==
                ReactorPlatformPollDisposition::Failed);
  static_assert(poll_failed.platform_error() == ENOMEM);

  constexpr ReactorPlatformPollResult poll_unavailable =
      ReactorPlatformPollResult::backend_unavailable();
  static_assert(poll_unavailable.disposition() ==
                ReactorPlatformPollDisposition::BackendUnavailable);
  static_assert(poll_unavailable.platform_error() == 0);

  // This contract is compiled against the neutral interface only. Platform
  // SDK records are deliberately unavailable at this boundary.

#if defined(__linux__)
  {
    int stale_pipe[2] = {-1, -1};
    int live_pipe[2] = {-1, -1};
    TEST_ASSERT(::pipe(stale_pipe) == 0);
    TEST_ASSERT(::pipe(live_pipe) == 0);
    ReactorPlatform invalid_remove_platform{};
    TEST_ASSERT(
        OperationSucceeded(PrepareReactorPlatform(invalid_remove_platform, 2u)));
    TEST_ASSERT(OperationSucceeded(OpenReactorPlatform(invalid_remove_platform)));
    const ReactorHandle stale_handle =
        ReactorHandleFromPublic(stale_pipe[0]);
    const ReactorHandle live_handle = ReactorHandleFromPublic(live_pipe[0]);
    TEST_ASSERT(OperationSucceeded(AddReactorPlatformInterest(
        invalid_remove_platform, stale_handle, ReactorInterest::Read)));
    TEST_ASSERT(OperationSucceeded(AddReactorPlatformInterest(
        invalid_remove_platform, live_handle, ReactorInterest::Read)));
    TEST_ASSERT(::close(stale_pipe[0]) == 0);
    stale_pipe[0] = -1;
    const ReactorPlatformOpResult invalid_remove =
        RemoveReactorPlatformInterest(invalid_remove_platform, stale_handle);
    TEST_ASSERT(invalid_remove.disposition() ==
                ReactorPlatformOpDisposition::Invalid);
    const char live_byte = 'l';
    TEST_ASSERT(::write(live_pipe[1], &live_byte, 1u) == 1);
    std::vector<ReactorPlatformReady> live_ready{};
    const ReactorPlatformPollResult live_poll = PollReactorPlatform(
        invalid_remove_platform, -1, 2u, live_ready);
    TEST_ASSERT(live_poll.disposition() ==
                ReactorPlatformPollDisposition::Success);
    TEST_ASSERT(live_ready.size() == 1u);
    TEST_ASSERT(live_ready.front().handle == live_handle);
    CloseReactorPlatform(invalid_remove_platform);
    TEST_ASSERT(::close(stale_pipe[1]) == 0);
    TEST_ASSERT(::close(live_pipe[0]) == 0);
    TEST_ASSERT(::close(live_pipe[1]) == 0);
  }
#endif

  ResetReactorBackendStats();
  ReactorPlatform platform{};

  TEST_ASSERT(OperationSucceeded(PrepareReactorPlatform(platform, 8u)));
  ReactorPlatformState* const storage = platform.state.get();
  TEST_ASSERT(storage != nullptr);
  TEST_ASSERT(OperationSucceeded(OpenReactorPlatform(platform)));
  CloseReactorPlatform(platform);

  TEST_ASSERT(OperationSucceeded(PrepareReactorPlatform(platform, 16u)));
  TEST_ASSERT(platform.state.get() == storage);
  TEST_ASSERT(OperationSucceeded(OpenReactorPlatform(platform)));
  CloseReactorPlatform(platform);

  {
    ReactorPlatform scoped{};
    TEST_ASSERT(OperationSucceeded(PrepareReactorPlatform(scoped, 4u)));
    TEST_ASSERT(OperationSucceeded(OpenReactorPlatform(scoped)));
  }

  int pipe_fds[2] = {-1, -1};
  TEST_ASSERT(::pipe(pipe_fds) == 0);
  const char byte = 'p';
  TEST_ASSERT(::write(pipe_fds[1], &byte, 1u) == 1);
  TEST_ASSERT(OperationSucceeded(OpenReactorPlatform(platform)));
  TEST_ASSERT(OperationSucceeded(AddReactorPlatformInterest(
      platform, ReactorHandleFromPublic(pipe_fds[0]), ReactorInterest::Read)));
  std::vector<ReactorPlatformReady> poll_ready{};
  const ReactorPlatformPollResult first_poll =
      PollReactorPlatform(platform, 0, 1u, poll_ready);
  TEST_ASSERT(first_poll.disposition() ==
              ReactorPlatformPollDisposition::Success);
  TEST_ASSERT(first_poll.platform_error() == 0);
  TEST_ASSERT(poll_ready.size() == 1u);
  TEST_ASSERT(HasReactorEvent(poll_ready.front().events, ReactorEvent::Read));

  runtime_task_allocation::Start();
  const ReactorPlatformPollResult warm_poll =
      PollReactorPlatform(platform, 0, 1u, poll_ready);
  runtime_task_allocation::Stop();
  TEST_ASSERT(warm_poll.disposition() ==
              ReactorPlatformPollDisposition::Success);
  TEST_ASSERT(warm_poll.platform_error() == 0);
  TEST_ASSERT(poll_ready.size() == 1u);
  TEST_ASSERT(runtime_task_allocation::Count() == 0u);

  char consumed = 0;
  TEST_ASSERT(::read(pipe_fds[0], &consumed, 1u) == 1);
  const ReactorHandle writable = ReactorHandleFromPublic(pipe_fds[1]);
  TEST_ASSERT(OperationSucceeded(AddReactorPlatformInterest(
      platform, writable, ReactorInterest::Write)));
  const ReactorPlatformPollResult writable_poll =
      PollReactorPlatform(platform, 0, 1u, poll_ready);
  TEST_ASSERT(writable_poll.disposition() ==
              ReactorPlatformPollDisposition::Success);
  TEST_ASSERT(writable_poll.platform_error() == 0);
  TEST_ASSERT(poll_ready.size() == 1u);
  TEST_ASSERT(poll_ready.front().handle == writable);
  TEST_ASSERT(HasReactorEvent(poll_ready.front().events, ReactorEvent::Write));
  TEST_ASSERT(OperationSucceeded(
      RemoveReactorPlatformInterest(platform, writable)));
  const ReactorPlatformPollResult removed_poll =
      PollReactorPlatform(platform, 0, 1u, poll_ready);
  TEST_ASSERT(removed_poll.disposition() ==
              ReactorPlatformPollDisposition::Success);
  TEST_ASSERT(removed_poll.platform_error() == 0);
  TEST_ASSERT(poll_ready.empty());
  TEST_ASSERT(::write(pipe_fds[1], &byte, 1u) == 1);
  CloseReactorPlatform(platform);

  std::vector<ReactorPlatformReady> cold_poll_ready{};
  cold_poll_ready.reserve(1u);
  const std::size_t poll_registration_count = cold_poll_ready.capacity() + 1u;
  std::vector<std::array<int, 2>> poll_pipes(poll_registration_count,
                                             std::array<int, 2>{-1, -1});
  ReactorPlatform poll_failure_platform{};
  TEST_ASSERT(OperationSucceeded(PrepareReactorPlatform(
      poll_failure_platform, poll_registration_count)));
  TEST_ASSERT(OperationSucceeded(OpenReactorPlatform(poll_failure_platform)));
  for (std::array<int, 2> &poll_pipe : poll_pipes) {
    TEST_ASSERT(::pipe(poll_pipe.data()) == 0);
    TEST_ASSERT(::write(poll_pipe[1], &byte, 1u) == 1);
    TEST_ASSERT(OperationSucceeded(AddReactorPlatformInterest(
        poll_failure_platform, ReactorHandleFromPublic(poll_pipe[0]),
        ReactorInterest::Read)));
  }
  runtime_task_allocation::FailNext();
  const ReactorPlatformPollResult poll_capacity_failed = PollReactorPlatform(
      poll_failure_platform, 0, poll_registration_count, cold_poll_ready);
  TEST_ASSERT(poll_capacity_failed.disposition() ==
              ReactorPlatformPollDisposition::Failed);
  TEST_ASSERT(poll_capacity_failed.platform_error() == ENOMEM);
  TEST_ASSERT(cold_poll_ready.empty());
  const ReactorPlatformPollResult poll_retried = PollReactorPlatform(
      poll_failure_platform, 0, poll_registration_count, cold_poll_ready);
  TEST_ASSERT(poll_retried.disposition() ==
              ReactorPlatformPollDisposition::Success);
  TEST_ASSERT(poll_retried.platform_error() == 0);
  TEST_ASSERT(cold_poll_ready.size() == poll_registration_count);
  runtime_task_allocation::Start();
  const ReactorPlatformPollResult poll_warm = PollReactorPlatform(
      poll_failure_platform, 0, poll_registration_count, cold_poll_ready);
  runtime_task_allocation::Stop();
  TEST_ASSERT(poll_warm.disposition() ==
              ReactorPlatformPollDisposition::Success);
  TEST_ASSERT(poll_warm.platform_error() == 0);
  TEST_ASSERT(cold_poll_ready.size() == poll_registration_count);
  TEST_ASSERT(runtime_task_allocation::Count() == 0u);
  CloseReactorPlatform(poll_failure_platform);
  for (const std::array<int, 2> &poll_pipe : poll_pipes) {
    TEST_ASSERT(::close(poll_pipe[0]) == 0);
    TEST_ASSERT(::close(poll_pipe[1]) == 0);
  }

  int duplex_fds[2] = {-1, -1};
  TEST_ASSERT(::socketpair(AF_UNIX, SOCK_STREAM, 0, duplex_fds) == 0);
  TEST_ASSERT(::write(duplex_fds[1], &byte, 1u) == 1);
  ReactorPlatform duplex_platform{};
  TEST_ASSERT(OperationSucceeded(PrepareReactorPlatform(duplex_platform, 1u)));
  TEST_ASSERT(OperationSucceeded(OpenReactorPlatform(duplex_platform)));
  TEST_ASSERT(OperationSucceeded(AddReactorPlatformInterest(
      duplex_platform, ReactorHandleFromPublic(duplex_fds[0]),
      ReactorInterest::Read | ReactorInterest::Write)));
  std::vector<ReactorPlatformReady> duplex_ready{};
  const ReactorPlatformPollResult duplex_poll =
      PollReactorPlatform(duplex_platform, 0, 1u, duplex_ready);
  TEST_ASSERT(duplex_poll.disposition() ==
              ReactorPlatformPollDisposition::Success);
  TEST_ASSERT(duplex_poll.platform_error() == 0);
  TEST_ASSERT(!duplex_ready.empty());
#if defined(__APPLE__) || defined(__FreeBSD__)
  TEST_ASSERT(duplex_ready.size() <= 2u);
#else
  TEST_ASSERT(duplex_ready.size() == 1u);
#endif
  ReactorEvent duplex_events = ReactorEvent::None;
  for (const ReactorPlatformReady &ready_event : duplex_ready) {
    duplex_events |= ready_event.events;
  }
  TEST_ASSERT(HasReactorEvent(duplex_events, ReactorEvent::Read));
  TEST_ASSERT(HasReactorEvent(duplex_events, ReactorEvent::Write));
  CloseReactorPlatform(duplex_platform);
  TEST_ASSERT(::close(duplex_fds[0]) == 0);
  TEST_ASSERT(::close(duplex_fds[1]) == 0);

  const BatchIoPollRequest request{
      .index = 0u,
      .handle = ReactorHandleFromPublic(pipe_fds[0]),
      .interest = ReactorInterest::Read,
  };
  std::vector<BatchIoReady> cold_ready{};
  cold_ready.reserve(1u);
  const std::size_t probe_count = cold_ready.capacity() + 1u;
  std::vector<BatchIoPollRequest> capacity_requests(probe_count, request);
  TEST_ASSERT(
      OperationSucceeded(PrepareReactorPlatform(platform, probe_count)));
  runtime_task_allocation::FailNext();
  const BatchIoProbeResult failed = ProbeReactorPlatformNow(
      platform, capacity_requests.data(), capacity_requests.size(), cold_ready);
  TEST_ASSERT(failed.disposition() == BatchIoProbeDisposition::Failed);
  TEST_ASSERT(failed.platform_error() == ENOMEM);
  TEST_ASSERT(cold_ready.empty());
  const BatchIoProbeResult retried = ProbeReactorPlatformNow(
      platform, capacity_requests.data(), capacity_requests.size(), cold_ready);
  TEST_ASSERT(retried.disposition() == BatchIoProbeDisposition::Success);
  TEST_ASSERT(retried.platform_error() == 0);
  TEST_ASSERT(!cold_ready.empty());

  std::vector<BatchIoReady> ready{};
  ready.reserve(1u);
  const BatchIoProbeResult first_probe =
      ProbeReactorPlatformNow(platform, &request, 1u, ready);
  TEST_ASSERT(first_probe.disposition() == BatchIoProbeDisposition::Success);
  TEST_ASSERT(first_probe.platform_error() == 0);
  TEST_ASSERT(ready.size() == 1u);

  runtime_task_allocation::Start();
  const BatchIoProbeResult warm =
      ProbeReactorPlatformNow(platform, &request, 1u, ready);
  runtime_task_allocation::Stop();
  TEST_ASSERT(warm.disposition() == BatchIoProbeDisposition::Success);
  TEST_ASSERT(warm.platform_error() == 0);
  TEST_ASSERT(ready.size() == 1u);
  TEST_ASSERT(runtime_task_allocation::Count() == 0u);
  TEST_ASSERT(::close(pipe_fds[0]) == 0);
  TEST_ASSERT(::close(pipe_fds[1]) == 0);

  const ReactorBackendStats stats = ReactorBackendStatsSnapshot();
  TEST_ASSERT(stats.open_calls == 6u);
  TEST_ASSERT(stats.close_calls == 6u);
  TEST_ASSERT(stats.current_open_handles == 0u);
  TEST_ASSERT(stats.max_open_handles == 1u);
  return 0;
}
