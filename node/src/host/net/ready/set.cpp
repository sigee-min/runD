#include <rund/net/ready/set.hpp>
#include <optional>

#include "../../../runtime/task/scheduler/access.hpp"
#include "../../../runtime/task/scheduler/state.hpp"
#include "../interest.hpp"
#include "../operation.hpp"
#include "../registry/socket.hpp"

namespace rund::net {
namespace {

[[nodiscard]] ready::Status FailReadySet(const ::rund::ReasonCode code) noexcept {
  return ready::Status{code};
}

[[nodiscard]] ready::many::Result FailReadySetWait(const ::rund::ReasonCode code) noexcept {
  return ready::many::Result{code};
}

} // namespace

ready::Status ready::create(const ready::Config options) noexcept {
  node::Scheduler *const scheduler = node::scheduler_access::ActiveScheduler();
  if (scheduler == nullptr) {
    return FailReadySet(::rund::ReasonCode::NodeRuntimeMissing);
  }
  return scheduler->CreateReadySet(options);
}

ready::Status ready::destroy(const ready::Set set) noexcept {
  node::Scheduler *const scheduler = node::scheduler_access::ActiveScheduler();
  if (scheduler == nullptr) {
    return FailReadySet(::rund::ReasonCode::NodeRuntimeMissing);
  }
  return scheduler->DestroyReadySet(set);
}

ready::Status ready::clear(const ready::Set set) noexcept {
  node::Scheduler *const scheduler = node::scheduler_access::ActiveScheduler();
  if (scheduler == nullptr) {
    return FailReadySet(::rund::ReasonCode::NodeRuntimeMissing);
  }
  return scheduler->ClearReadySet(set);
}

ready::Status ready::add(const ready::Set set,
                                  const ready::Request request) noexcept {
  const node::ReactorInterest reactor_interest = ReactorInterestFor(request.interest);
  if (reactor_interest == node::ReactorInterest::None) {
    return FailReadySet(::rund::ReasonCode::TaskInvalid);
  }
  SocketLease lease = LeaseSocket(request.socket);
  if (!lease) {
    return FailReadySet(::rund::ReasonCode::IoFdInvalid);
  }
  node::Scheduler *const scheduler = node::scheduler_access::ActiveScheduler();
  if (scheduler == nullptr) {
    return FailReadySet(::rund::ReasonCode::NodeRuntimeMissing);
  }
  return scheduler->AddReadyInterest(set, request);
}

ready::Status ready::remove(const ready::Set set,
                                     const ready::Request request) noexcept {
  const node::ReactorInterest reactor_interest = ReactorInterestFor(request.interest);
  if (reactor_interest == node::ReactorInterest::None) {
    return FailReadySet(::rund::ReasonCode::TaskInvalid);
  }
  SocketLease lease = LeaseSocket(request.socket);
  if (!lease) {
    return FailReadySet(::rund::ReasonCode::IoFdInvalid);
  }
  node::Scheduler *const scheduler = node::scheduler_access::ActiveScheduler();
  if (scheduler == nullptr) {
    return FailReadySet(::rund::ReasonCode::NodeRuntimeMissing);
  }
  return scheduler->RemoveReadyInterest(set, request);
}

ready::many::Wait ready::many::wait(const ready::Set set, const std::span<ready::Event> out,
                       const ready::many::Budget budget) noexcept {
  if (budget.max_events == 0u) {
    ready::many::Result result{::rund::ReasonCode::Ok};
    result.budget_exhausted = true;
    return ready::many::detail::Access::Complete(result);
  }
  node::Scheduler *const scheduler = node::scheduler_access::ActiveScheduler();
  if (scheduler == nullptr) {
    return ready::many::detail::Access::Complete(
        FailReadySetWait(::rund::ReasonCode::NodeRuntimeMissing));
  }
  if (scheduler->CurrentTaskIsCoroutine()) {
    return ready::many::detail::Access::Defer(
        {}, out, budget, 0, false, 0u, 0u, 0u, 0u, set.id, set.generation);
  }
  return scheduler->WaitReadySet(set, out, std::nullopt, budget);
}

ready::many::Wait ready::many::wait(const ready::Set set, const std::span<ready::Event> out,
                           const std::chrono::nanoseconds timeout,
                           const ready::many::Budget budget) noexcept {
  if (timeout.count() < 0) {
    return ready::many::detail::Access::Complete(
        FailReadySetWait(::rund::ReasonCode::TimerDurationInvalid));
  }
  if (budget.max_events == 0u) {
    ready::many::Result result{::rund::ReasonCode::Ok};
    result.budget_exhausted = true;
    return ready::many::detail::Access::Complete(result);
  }
  node::Scheduler *const scheduler = node::scheduler_access::ActiveScheduler();
  if (scheduler == nullptr) {
    return ready::many::detail::Access::Complete(
        FailReadySetWait(::rund::ReasonCode::NodeRuntimeMissing));
  }
  if (scheduler->CurrentTaskIsCoroutine()) {
    return ready::many::detail::Access::Defer(
        {}, out, budget, timeout.count(), true, 0u, 0u, 0u, 0u, set.id,
        set.generation);
  }
  return scheduler->WaitReadySet(set, out, timeout, budget);
}

} // namespace rund::net
