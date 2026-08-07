#include "../lease.hpp"
#include "local.hpp"

namespace rund::node {

::rund::net::ready::many::Wait Scheduler::WaitReactorMany(
    const std::span<const ::rund::net::ready::Request> requests,
    const std::span<::rund::net::ready::Event> out,
    const std::optional<std::chrono::nanoseconds> timeout,
    const ::rund::net::ready::many::Budget budget,
    const ::rund::detail::task::StopIdentity stop) noexcept {
  if (requests.size() > state_->resources.limits.reactor_wait_capacity ||
      requests.size() >
          state_->reactor.reactor_socket_lease_scratch.capacity()) {
    return FailManyCode(ReasonCode::ReactorWaitCapacityExceeded);
  }
  ReactorLeaseScope leases{state_->reactor.reactor_socket_lease_scratch};
  if (!leases.acquire(requests, [](const auto &request) noexcept {
        return ReactorLeaseSource::socket(request.socket);
      })) {
    return FailManyCode(ReasonCode::IoFdInvalid);
  }
  if (!ReactorManyBuildRequests(requests, leases.values(),
                                state_->reactor.reactor_many_request_scratch)) {
    return FailManyCode(ReasonCode::ReactorWaitCapacityExceeded);
  }
  return WaitReactorManyPrepared(
      state_->reactor.reactor_many_request_scratch, out, timeout, budget,
      stop, {});
}

::rund::net::ready::many::Wait Scheduler::WaitReactorManyPrepared(
    const std::span<const ReactorManyRequest> requests,
    const std::span<::rund::net::ready::Event> out,
    const std::optional<std::chrono::nanoseconds> timeout,
    const ::rund::net::ready::many::Budget budget,
    const ::rund::detail::task::StopIdentity stop,
    const ::rund::net::ready::Set ready_set) noexcept {
  ReadyManyEntry entry = ReadyManyAccess::PrepareEntry(
      *this, requests, out, budget, stop);
  if (!entry.ok()) {
    return FailManyCode(entry.code);
  }

  ::rund::net::ready::many::Wait immediate =
      ReadyManyAccess::TryImmediate(*this, entry, out, timeout);
  if (::rund::net::ready::many::detail::Access::ResultOf(immediate).code() !=
      ReasonCode::NetReadyManyNotReady) {
    return immediate;
  }

  ::rund::net::ready::many::Wait parked =
      ReadyManyAccess::Park(*this, entry, timeout, ready_set);
  if (::rund::net::ready::many::detail::Access::ResultOf(parked).code() !=
          ReasonCode::NetReadyManyNotReady ||
      ::rund::net::ready::many::detail::Access::Suspended(parked)) {
    return parked;
  }

  return ReadyManyAccess::Resume(*this, entry, out, entry.group_id);
}

::rund::net::ready::many::Wait
Scheduler::ResumeReactorMany(const std::span<::rund::net::ready::Event> out,
                             const std::uint64_t group_id) noexcept {
  ReadyManyEntry entry{
      .record = state_->Find(CurrentTaskId()),
      .task_id = CurrentTaskId(),
      .group_id = group_id,
      .code = ReasonCode::Ok,
  };
  return ReadyManyAccess::Resume(*this, entry, out, group_id);
}

} // namespace rund::node
