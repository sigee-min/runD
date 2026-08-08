#include "../../../../../src/runtime/reactor/readiness/handle.hpp"
#include "../../../../../src/runtime/reactor/readiness/mask.hpp"
#include "../../../../../src/runtime/task/scheduler/reactor/backend.hpp"
#include "../../../../../src/runtime/task/scheduler/reactor/lease.hpp"
#include "../../../../../src/runtime/task/scheduler/reactor/many/events.hpp"
#include "../../../../../src/runtime/task/scheduler/reactor/many/probe/raw.hpp"
#include "../../../../../src/runtime/task/scheduler/reactor/model.hpp"
#include "../../../../../src/runtime/task/scheduler/reactor/poll.hpp"

#include "../coroutine/allocation.hpp"
#include "test/assert.hpp"

#include <array>
#include <cerrno>
#include <limits>
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
  static_assert(!std::is_default_constructible_v<ReactorLeaseSource>);
  static_assert(!std::is_aggregate_v<ReactorLeaseSource>);
  static_assert(std::is_trivially_copyable_v<ReactorLeaseSource>);
  static_assert(!std::is_aggregate_v<ReactorManyEventSlot>);
  static_assert(!std::is_aggregate_v<ReactorManyProbeResult>);
  static_assert(!std::is_default_constructible_v<ReactorManyProbeResult>);
  static_assert(std::is_trivially_copyable_v<ReactorManyProbeResult>);

  constexpr ReactorLeaseSource invalid_lease_source =
      ReactorLeaseSource::invalid();
  static_assert(invalid_lease_source.disposition() ==
                ReactorLeaseSourceDisposition::Invalid);
  static_assert(!invalid_lease_source.socket_view());
  constexpr ReactorLeaseSource host_lease_source =
      ReactorLeaseSource::host_fd();
  static_assert(host_lease_source.disposition() ==
                ReactorLeaseSourceDisposition::HostFd);
  static_assert(!host_lease_source.socket_view());
  constexpr ReactorLeaseSource empty_socket_lease_source =
      ReactorLeaseSource::socket({});
  static_assert(empty_socket_lease_source.disposition() ==
                ReactorLeaseSourceDisposition::Invalid);
  static_assert(!empty_socket_lease_source.socket_view());

  const std::array lease_inputs{1u};
  std::vector<::rund::net::SocketLease> lease_storage{};
  lease_storage.reserve(lease_inputs.size());
  {
    ReactorLeaseScope host_leases{lease_storage};
    TEST_ASSERT(host_leases.acquire(lease_inputs, [](const auto) noexcept {
      return ReactorLeaseSource::host_fd();
    }));
    TEST_ASSERT(host_leases.values().empty());
  }
  {
    ReactorLeaseScope invalid_leases{lease_storage};
    TEST_ASSERT(!invalid_leases.acquire(lease_inputs, [](const auto) noexcept {
      return ReactorLeaseSource::invalid();
    }));
    TEST_ASSERT(invalid_leases.values().empty());
  }
  {
    ReactorLeaseScope empty_socket_leases{lease_storage};
    TEST_ASSERT(
        !empty_socket_leases.acquire(lease_inputs, [](const auto) noexcept {
          return ReactorLeaseSource::socket({});
        }));
    TEST_ASSERT(empty_socket_leases.values().empty());
  }

  constexpr ReactorManyProbeResult many_probe_success =
      ReactorManyProbeResult::success(3u);
  static_assert(many_probe_success.ok());
  static_assert(many_probe_success.code() == ::rund::ReasonCode::Ok);
  static_assert(many_probe_success.total_ready() == 3u);

  constexpr ReactorManyProbeResult many_probe_capacity =
      ReactorManyProbeResult::wait_capacity_exceeded();
  static_assert(!many_probe_capacity.ok());
  static_assert(many_probe_capacity.code() ==
                ::rund::ReasonCode::ReactorWaitCapacityExceeded);
  static_assert(many_probe_capacity.total_ready() == 0u);

  constexpr ReactorManyProbeResult many_probe_unavailable =
      ReactorManyProbeResult::backend_unavailable();
  static_assert(!many_probe_unavailable.ok());
  static_assert(many_probe_unavailable.code() ==
                ::rund::ReasonCode::ReactorBackendUnavailable);
  static_assert(many_probe_unavailable.total_ready() == 0u);

  constexpr ReactorManyProbeResult many_probe_failed =
      ReactorManyProbeResult::poll_failed();
  static_assert(!many_probe_failed.ok());
  static_assert(many_probe_failed.code() == ::rund::ReasonCode::IoPollFailed);
  static_assert(many_probe_failed.total_ready() == 0u);

  constexpr ReactorManyProbeResult many_probe_mismatch =
      ReactorManyProbeResult::host_replay_mismatch(5u);
  static_assert(!many_probe_mismatch.ok());
  static_assert(many_probe_mismatch.code() ==
                ::rund::ReasonCode::HostReplayEventMismatch);
  static_assert(many_probe_mismatch.total_ready() == 5u);

  constexpr ReactorManyProbeResult many_probe_invalid =
      ReactorManyProbeResult::invalid_after(7u);
  static_assert(!many_probe_invalid.ok());
  static_assert(many_probe_invalid.code() == ::rund::ReasonCode::IoFdInvalid);
  static_assert(many_probe_invalid.total_ready() == 8u);

  constexpr ReactorManyProbeResult many_probe_invalid_overflow =
      ReactorManyProbeResult::invalid_after(
          std::numeric_limits<std::uint32_t>::max());
  static_assert(!many_probe_invalid_overflow.ok());
  static_assert(many_probe_invalid_overflow.code() ==
                ::rund::ReasonCode::ReactorWaitCapacityExceeded);
  static_assert(many_probe_invalid_overflow.total_ready() == 0u);

  ReactorManyGroup event_group{
      .group_id = 41u,
      .request_count = 2u,
      .max_events = 1u,
  };
  std::vector<ReactorManyEventSlot> event_slots{};
  ReactorManyEventSlotsReset(event_slots, event_group.request_count);
  TEST_ASSERT(event_slots.size() == event_group.request_count);
  TEST_ASSERT(!event_slots[0].has_value());
  TEST_ASSERT(!event_slots[1].has_value());

  const ReactorManyRequest first_event_request{
      .group_id = event_group.group_id,
      .fd = ReactorHandleFromPublic(13),
      .slot = 0u,
      .event_index = 17u,
      .interest = ReactorInterest::Read,
  };
  const ReactorManyRequest stale_event_request{
      .group_id = event_group.group_id + 1u,
      .fd = first_event_request.fd,
      .slot = first_event_request.slot,
      .event_index = first_event_request.event_index,
      .interest = first_event_request.interest,
  };
  ReactorManyEventSlotsAppend(event_group, stale_event_request,
                              ReactorEvent::Read, ::rund::ReasonCode::Ok,
                              event_slots);
  TEST_ASSERT(!event_slots[0].has_value());
  TEST_ASSERT(event_group.stored_event_count == 0u);
  TEST_ASSERT(!event_group.budget_exhausted);

  const ReactorManyRequest out_of_range_event_request{
      .group_id = event_group.group_id,
      .fd = first_event_request.fd,
      .slot = event_group.request_count,
      .event_index = first_event_request.event_index,
      .interest = first_event_request.interest,
  };
  ReactorManyEventSlotsAppend(event_group, out_of_range_event_request,
                              ReactorEvent::Read, ::rund::ReasonCode::Ok,
                              event_slots);
  TEST_ASSERT(!event_slots[0].has_value());
  TEST_ASSERT(!event_slots[1].has_value());
  TEST_ASSERT(event_group.stored_event_count == 0u);
  TEST_ASSERT(!event_group.budget_exhausted);

  ReactorManyEventSlotsAppend(event_group, first_event_request,
                              ReactorEvent::Read, ::rund::ReasonCode::Ok,
                              event_slots);
  TEST_ASSERT(event_slots[0].has_value());
  TEST_ASSERT(event_slots[0]->group_id == event_group.group_id);
  TEST_ASSERT(event_slots[0]->fd == first_event_request.fd);
  TEST_ASSERT(event_slots[0]->event_index == first_event_request.event_index);
  TEST_ASSERT(event_slots[0]->events == ReactorEvent::Read);
  TEST_ASSERT(event_group.stored_event_count == 1u);

  ReactorManyEventSlotsAppend(event_group, first_event_request,
                              ReactorEvent::Error,
                              ::rund::ReasonCode::IoFdInvalid, event_slots);
  TEST_ASSERT(event_slots[0]->events == ReactorEvent::Read);
  TEST_ASSERT(event_slots[0]->code == ::rund::ReasonCode::Ok);
  TEST_ASSERT(event_group.stored_event_count == 1u);
  TEST_ASSERT(!event_group.budget_exhausted);

  const ReactorManyRequest budgeted_event_request{
      .group_id = event_group.group_id,
      .fd = ReactorHandleFromPublic(19),
      .slot = 1u,
      .event_index = 23u,
      .interest = ReactorInterest::Write,
  };
  ReactorManyEventSlotsAppend(event_group, budgeted_event_request,
                              ReactorEvent::Write, ::rund::ReasonCode::Ok,
                              event_slots);
  TEST_ASSERT(!event_slots[1].has_value());
  TEST_ASSERT(event_group.stored_event_count == 1u);
  TEST_ASSERT(event_group.budget_exhausted);

  std::array<::rund::net::ready::Event, 2u> copied_events{};
  std::uint32_t copied_event_count = 0u;
  TEST_ASSERT(ReactorManyEventSlotsCopy(event_group, event_slots, copied_events,
                                        &copied_event_count));
  TEST_ASSERT(copied_event_count == 1u);
  TEST_ASSERT(copied_events[0].index == first_event_request.event_index);
  TEST_ASSERT(copied_events[0].ticket.code() == ::rund::ReasonCode::Ok);

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
