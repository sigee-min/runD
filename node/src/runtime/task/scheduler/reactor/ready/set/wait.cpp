#include "operations.hpp"

#include "../../../../../../host/net/ready/ticket.hpp"
#include "../../../../../../host/net/socket/access.hpp"
#include "../../../../../reactor/readiness/handle.hpp"
#include "../../../../../reactor/readiness/mask.hpp"
#include "../../lease.hpp"

#include <optional>
#include <utility>
#include <vector>

namespace rund::node {

::rund::net::ready::many::Wait Scheduler::WaitReadySet(
    const ::rund::net::ready::Set handle,
    const std::span<::rund::net::ready::Event> out,
    const std::optional<std::chrono::nanoseconds> timeout,
    const ::rund::net::ready::many::Budget budget) noexcept {
  if (budget.max_events == 0u) {
    return ::rund::net::ready::many::detail::Access::Complete(
        ReadySetWaitStatus(ReasonCode::Ok, 0u, true));
  }

  (void)TrapLaneOwnedSegmentPrimitive(
      ::rund::detail::task::OperationKind::IoPark);
  EnsureCurrentCommit();
  const ReactorReadySet *const set =
      ReactorReadySetFind(state_->reactor.reactor_ready_sets, handle);
  if (set == nullptr) {
    ::rund::net::ready::many::Result result =
        ReadySetWaitStatus(ReasonCode::TaskInvalid);
    CompletePrimitiveCommit();
    return ::rund::net::ready::many::detail::Access::Complete(result);
  }

  if (set->members.empty()) {
    ::rund::net::ready::many::Result result =
        ReadySetWaitStatus(ReasonCode::Ok);
    CompletePrimitiveCommit();
    return ::rund::net::ready::many::detail::Access::Complete(result);
  }
  if (out.empty()) {
    ::rund::net::ready::many::Result result =
        ReadySetWaitStatus(ReasonCode::TaskInvalid);
    CompletePrimitiveCommit();
    return ::rund::net::ready::many::detail::Access::Complete(result);
  }

  std::vector<ReactorManyRequest> &requests =
      state_->reactor.reactor_many_request_scratch;
  requests.clear();
  try {
    requests.resize(set->members.size());
    for (std::size_t slot = 0u; slot < set->members.size(); ++slot) {
      const ReactorReadySetMember &member = set->members[slot];
      requests[slot] = ReactorManyRequest{
          .socket = member.socket,
          .fd = member.fd,
          .slot = static_cast<std::uint32_t>(slot),
          .event_index = member.index,
          .interest = member.interest,
      };
    }
  } catch (...) {
    ::rund::net::ready::many::Result result =
        ReadySetWaitStatus(ReasonCode::ReactorWaitCapacityExceeded);
    CompletePrimitiveCommit();
    return ::rund::net::ready::many::detail::Access::Complete(result);
  }

  RecordReactorReadySetWait(state_->evidence.metrics);
  for (const ReactorReadySetMember &member : set->members) {
    if (ReadySetMemberIsCurrent(member)) {
      continue;
    }
    ::rund::net::ready::Interest interest{};
    if (!::rund::net::InterestFromReactor(member.interest, &interest)) {
      ::rund::net::ready::many::Result result =
          ReadySetWaitStatus(ReasonCode::TaskInvalid);
      CompletePrimitiveCommit();
      return ::rund::net::ready::many::detail::Access::Complete(result);
    }
    ::rund::net::ready::Event event{};
    event.index = member.index;
    event.ticket = ::rund::net::ready::detail::Access::make(
        ReasonCode::IoFdInvalid, member.socket, interest,
        ReactorEventBits(ReactorEventsForInterest(member.interest)));
    out[0u] = std::move(event);
    const std::uint64_t task_id = CurrentTaskId();
    RecordReactorObservation(
        task::ObservationKind::IoInvalid, ReasonCode::IoFdInvalid, task_id, 0u,
        ReactorHandleForPublic(member.fd), ReactorInterestBits(member.interest),
        ReactorEventBits(ReactorEventsForInterest(member.interest)));
    if (!RecordReactorHostEvent(ReasonCode::IoFdInvalid, task_id,
                                ::rund::net::detail::SocketAccess::id(
                                    ReactorHandleForPublic(member.fd)))) {
      ::rund::net::ready::many::Result result =
          ReadySetWaitStatus(ReasonCode::HostReplayEventMismatch, 1u);
      CompletePrimitiveCommit();
      return ::rund::net::ready::many::detail::Access::Complete(result);
    }
    RecordReactorReadySetInvalidations(state_->evidence.metrics, 1u);
    RecordReactorReadySetEvents(state_->evidence.metrics, 1u);
    ::rund::net::ready::many::Result result =
        ReadySetWaitStatus(ReasonCode::IoFdInvalid, 1u);
    CompletePrimitiveCommit();
    return ::rund::net::ready::many::detail::Access::Complete(result);
  }

  ReactorLeaseScope leases{state_->reactor.reactor_socket_lease_scratch};
  if (!leases.acquire(set->members, [](const auto &member) noexcept {
        return ReactorLeaseSource::socket(member.socket);
      })) {
    ::rund::net::ready::many::Result result =
        ReadySetWaitStatus(ReasonCode::IoFdInvalid);
    CompletePrimitiveCommit();
    return ::rund::net::ready::many::detail::Access::Complete(result);
  }

  ::rund::net::ready::many::Wait result = WaitReactorManyPrepared(
      requests, out, timeout, budget, 0u, 0u, 0u, 0u, set->identity.handle);
  EnsureCurrentCommit();
  RecordReactorReadySetEvents(
      state_->evidence.metrics,
      ::rund::net::ready::many::detail::Access::ResultOf(result).events);
  CompletePrimitiveCommit();
  return result;
}

} // namespace rund::node
