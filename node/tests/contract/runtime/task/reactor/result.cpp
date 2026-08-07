#include "../../../../../src/runtime/reactor/readiness/handle.hpp"
#include "../../../../../src/runtime/reactor/readiness/mask.hpp"
#include "../../../../../src/runtime/task/scheduler/reactor/backend.hpp"
#include "../../../../../src/runtime/task/scheduler/reactor/many/probe/raw.hpp"
#include "../../../../../src/runtime/task/scheduler/reactor/model.hpp"
#include "../../../../../src/runtime/task/scheduler/reactor/poll.hpp"

#include "../coroutine/allocation.hpp"
#include "test/assert.hpp"

#include <array>
#include <cerrno>
#include <type_traits>
#include <unistd.h>
#include <vector>

int RunRuntimeTaskReactorResultContract() {
  using namespace rund::node;

  static_assert(!std::is_aggregate_v<ReactorProbeResult>);
  static_assert(std::is_trivially_copyable_v<ReactorProbeResult>);
  static_assert(!std::is_aggregate_v<ReactorApplyResult>);
  static_assert(std::is_trivially_copyable_v<ReactorApplyResult>);
  static_assert(!std::is_aggregate_v<ReactorInvalidChangeToken>);
  static_assert(std::is_trivially_copyable_v<ReactorInvalidChangeToken>);

  constexpr ReactorApplyResult apply_success = ReactorApplyResult::success();
  static_assert(apply_success.disposition() ==
                ReactorApplyDisposition::Success);
  static_assert(!apply_success.invalid_change().valid());
  static_assert(ReactorApplyAllowsLogicalProgress(apply_success));

  constexpr ReactorHandle invalid_apply_handle = ReactorHandleFromPublic(17);
  constexpr std::uint64_t invalid_apply_generation = 19u;
  constexpr ReactorApplyResult apply_invalid =
      ReactorApplyResult::invalid(invalid_apply_handle,
                                  invalid_apply_generation);
  static_assert(apply_invalid.disposition() ==
                ReactorApplyDisposition::Invalid);
  static_assert(apply_invalid.invalid_change().handle() ==
                invalid_apply_handle);
  static_assert(apply_invalid.invalid_change().fd_generation() ==
                invalid_apply_generation);
  static_assert(ReactorApplyAllowsLogicalProgress(apply_invalid));

  constexpr ReactorApplyResult missing_invalid =
      ReactorApplyResult::invalid(kInvalidReactorHandle, 23u);
  static_assert(missing_invalid.disposition() ==
                ReactorApplyDisposition::Failed);
  static_assert(!missing_invalid.invalid_change().valid());
  static_assert(missing_invalid.invalid_change().handle() ==
                kInvalidReactorHandle);
  static_assert(missing_invalid.invalid_change().fd_generation() == 0u);
  static_assert(!ReactorApplyAllowsLogicalProgress(missing_invalid));

  constexpr ReactorApplyResult apply_failed = ReactorApplyResult::failed();
  static_assert(apply_failed.disposition() == ReactorApplyDisposition::Failed);
  static_assert(!apply_failed.invalid_change().valid());
  static_assert(!ReactorApplyAllowsLogicalProgress(apply_failed));

  constexpr ReactorApplyResult apply_unavailable =
      ReactorApplyResult::backend_unavailable();
  static_assert(apply_unavailable.disposition() ==
                ReactorApplyDisposition::BackendUnavailable);
  static_assert(!apply_unavailable.invalid_change().valid());
  static_assert(!ReactorApplyAllowsLogicalProgress(apply_unavailable));

  constexpr ReactorReady ready{};
  constexpr ReactorReady invalid_ready{
      .disposition = ReactorReadyDisposition::Invalid,
  };
  constexpr ReactorReady failed_ready{
      .disposition = ReactorReadyDisposition::PollFailed,
  };
  static_assert(ready.disposition == ReactorReadyDisposition::Ready);
  static_assert(invalid_ready.disposition == ReactorReadyDisposition::Invalid);
  static_assert(failed_ready.disposition ==
                ReactorReadyDisposition::PollFailed);

  constexpr ReactorProbeResult not_ready = ReactorProbeResult::not_ready();
  static_assert(not_ready.disposition() == ReactorProbeDisposition::NotReady);
  static_assert(not_ready.events() == ReactorEvent::None);

  constexpr ReactorEvent ready_events =
      ReactorEvent::Read | ReactorEvent::Hangup;
  constexpr ReactorProbeResult ready_result =
      ReactorProbeResult::ready(ready_events);
  static_assert(ready_result.disposition() == ReactorProbeDisposition::Ready);
  static_assert(ready_result.events() == ready_events);

  constexpr ReactorEvent invalid_events =
      ReactorEvent::Error | ReactorEvent::Hangup;
  constexpr ReactorProbeResult invalid_result =
      ReactorProbeResult::invalid(invalid_events);
  static_assert(invalid_result.disposition() ==
                ReactorProbeDisposition::Invalid);
  static_assert(invalid_result.events() == invalid_events);

  constexpr ReactorProbeResult poll_failed = ReactorProbeResult::poll_failed();
  static_assert(poll_failed.disposition() ==
                ReactorProbeDisposition::PollFailed);
  static_assert(poll_failed.events() == ReactorEvent::None);

  constexpr ReactorProbeResult backend_unavailable =
      ReactorProbeResult::backend_unavailable();
  static_assert(backend_unavailable.disposition() ==
                ReactorProbeDisposition::BackendUnavailable);
  static_assert(backend_unavailable.events() == ReactorEvent::None);

  ReactorPlatform platform{};
  TEST_ASSERT(PrepareReactorPlatform(platform, 1u).disposition() ==
              ReactorPlatformOpDisposition::Success);
  int pipe_fds[2] = {-1, -1};
  TEST_ASSERT(::pipe(pipe_fds) == 0);
  const ReactorHandle read_handle = ReactorHandleFromPublic(pipe_fds[0]);
  std::vector<BatchIoReady> scratch{};

  const ReactorProbeResult empty =
      ReactorProbeNow(platform, scratch, read_handle, ReactorInterest::Read);
  TEST_ASSERT(empty.disposition() == ReactorProbeDisposition::NotReady);
  TEST_ASSERT(empty.events() == ReactorEvent::None);

  const char byte = 'r';
  TEST_ASSERT(::write(pipe_fds[1], &byte, 1u) == 1);
  std::vector<BatchIoReady> cold_scratch{};
  runtime_task_allocation::FailNext();
  const ReactorProbeResult failed = ReactorProbeNow(
      platform, cold_scratch, read_handle, ReactorInterest::Read);
  TEST_ASSERT(failed.disposition() == ReactorProbeDisposition::PollFailed);
  TEST_ASSERT(failed.events() == ReactorEvent::None);

  const ReactorProbeResult readable = ReactorProbeNow(
      platform, cold_scratch, read_handle, ReactorInterest::Read);
  TEST_ASSERT(readable.disposition() == ReactorProbeDisposition::Ready);
  TEST_ASSERT(readable.events() == ReactorEvent::Read);

  std::vector<BatchIoPollRequest> poll_requests{};
  poll_requests.reserve(1u);
  const std::size_t request_count = poll_requests.capacity() + 1u;
  std::vector<ReactorManyRequest> many_requests(request_count);
  std::vector<std::array<int, 2>> extra_pipes(request_count - 1u,
                                              std::array<int, 2>{-1, -1});
  for (std::size_t index = 0u; index < many_requests.size(); ++index) {
    ReactorHandle handle = read_handle;
    if (index != 0u) {
      std::array<int, 2> &extra = extra_pipes[index - 1u];
      TEST_ASSERT(::pipe(extra.data()) == 0);
      TEST_ASSERT(::write(extra[1], &byte, 1u) == 1);
      handle = ReactorHandleFromPublic(extra[0]);
    }
    many_requests[index] = ReactorManyRequest{
        .fd = handle,
        .slot = static_cast<std::uint32_t>(index),
        .interest = ReactorInterest::Read,
    };
  }
  std::vector<BatchIoReady> many_ready{BatchIoReady{}};
  runtime_task_allocation::FailNext();
  const BatchIoProbeResult many_failed = ReactorProbeManyReadyNow(
      platform, many_requests, poll_requests, many_ready);
  TEST_ASSERT(many_failed.disposition() == BatchIoProbeDisposition::Failed);
  TEST_ASSERT(many_failed.platform_error() == ENOMEM);
  TEST_ASSERT(poll_requests.empty());
  TEST_ASSERT(many_ready.empty());

  const BatchIoProbeResult many_retried = ReactorProbeManyReadyNow(
      platform, many_requests, poll_requests, many_ready);
  TEST_ASSERT(many_retried.disposition() == BatchIoProbeDisposition::Success);
  TEST_ASSERT(many_retried.platform_error() == 0);
  TEST_ASSERT(poll_requests.size() == request_count);
  TEST_ASSERT(many_ready.size() == request_count);
  for (const std::array<int, 2> &extra : extra_pipes) {
    TEST_ASSERT(::close(extra[0]) == 0);
    TEST_ASSERT(::close(extra[1]) == 0);
  }

  char consumed = 0;
  TEST_ASSERT(::read(pipe_fds[0], &consumed, 1u) == 1);
  TEST_ASSERT(::close(pipe_fds[0]) == 0);
  pipe_fds[0] = -1;
  const ReactorProbeResult invalid = ReactorProbeNow(
      platform, cold_scratch, read_handle, ReactorInterest::Read);
  TEST_ASSERT(invalid.disposition() == ReactorProbeDisposition::Invalid);
  TEST_ASSERT(invalid.events() == ReactorEvent::None);
  TEST_ASSERT(::close(pipe_fds[1]) == 0);
  return 0;
}
