#include <algorithm>
#include <rund/net/cancel.hpp>

#include "../../runtime/platform/io.hpp"
#include "../../runtime/reactor/readiness/mask.hpp"
#include "../../runtime/task/scheduler/access.hpp"
#include "../../runtime/task/scheduler/state.hpp"
#include "interest.hpp"
#include "operation.hpp"
#include "ready/ticket.hpp"
#include "ready/validation.hpp"
#include "registry/socket.hpp"
#include "scheduler.hpp"

namespace rund::net {
namespace {

[[nodiscard]] ready::many::Result
FailReadyMany(const ::rund::ReasonCode code) noexcept {
  return ready::many::Result{code};
}

[[nodiscard]] ready::Ticket
TimedReadyFromIo(const ::rund::detail::task::IoDecision result,
                 const SocketView socket,
                 const ready::Interest interest) noexcept {
  return ready::detail::Access::make(result.status.code(), socket, interest,
                                     result.revents);
}

} // namespace

ready::timed::Wait ready::timed::read(const SocketView socket,
                                      const std::chrono::nanoseconds timeout,
                                      const task::stop_token token) noexcept {
  if (timeout.count() < 0) {
    return ready::timed::detail::Access::Complete(
        ready::detail::Access::make(::rund::ReasonCode::TimerDurationInvalid,
                                    socket, ready::Interest::Readable));
  }
  std::uint64_t scheduler_id = 0u;
  std::uint64_t source_id = 0u;
  std::uint64_t generation = 0u;
  std::uint64_t epoch = 0u;
  if (!node::scheduler_access::StopTokenIdentity(
          token, &scheduler_id, &source_id, &generation, &epoch)) {
    return ready::timed::detail::Access::Complete(ready::detail::Access::make(
        ::rund::ReasonCode::TaskInvalid, socket, ready::Interest::Readable));
  }
  node::Scheduler *const scheduler = node::scheduler_access::ActiveScheduler();
  if (scheduler == nullptr) {
    const ::rund::ReasonCode code = IsCurrentSocket(socket)
                                        ? ::rund::ReasonCode::NodeRuntimeMissing
                                        : ::rund::ReasonCode::IoFdInvalid;
    return ready::timed::detail::Access::Complete(
        ready::detail::Access::make(code, socket, ready::Interest::Readable));
  }
  if (scheduler->CurrentTaskIsCoroutine()) {
    return ready::timed::detail::Access::Defer(
        socket, ready::Interest::Readable,
        node::ReactorInterestBits(node::ReactorInterest::Read), timeout.count(),
        scheduler_id, source_id, generation, epoch);
  }
  return ready::timed::detail::Access::Complete(TimedReadyFromIo(
      WaitReactorTimed(*scheduler, socket,
                       node::ReactorInterestBits(node::ReactorInterest::Read),
                       timeout, scheduler_id, source_id, generation, epoch),
      socket, ready::Interest::Readable));
}

ready::timed::Wait ready::timed::write(const SocketView socket,
                                       const std::chrono::nanoseconds timeout,
                                       const task::stop_token token) noexcept {
  if (timeout.count() < 0) {
    return ready::timed::detail::Access::Complete(
        ready::detail::Access::make(::rund::ReasonCode::TimerDurationInvalid,
                                    socket, ready::Interest::Writable));
  }
  std::uint64_t scheduler_id = 0u;
  std::uint64_t source_id = 0u;
  std::uint64_t generation = 0u;
  std::uint64_t epoch = 0u;
  if (!node::scheduler_access::StopTokenIdentity(
          token, &scheduler_id, &source_id, &generation, &epoch)) {
    return ready::timed::detail::Access::Complete(ready::detail::Access::make(
        ::rund::ReasonCode::TaskInvalid, socket, ready::Interest::Writable));
  }
  node::Scheduler *const scheduler = node::scheduler_access::ActiveScheduler();
  if (scheduler == nullptr) {
    const ::rund::ReasonCode code = IsCurrentSocket(socket)
                                        ? ::rund::ReasonCode::NodeRuntimeMissing
                                        : ::rund::ReasonCode::IoFdInvalid;
    return ready::timed::detail::Access::Complete(
        ready::detail::Access::make(code, socket, ready::Interest::Writable));
  }
  if (scheduler->CurrentTaskIsCoroutine()) {
    return ready::timed::detail::Access::Defer(
        socket, ready::Interest::Writable,
        node::ReactorInterestBits(node::ReactorInterest::Write),
        timeout.count(), scheduler_id, source_id, generation, epoch);
  }
  return ready::timed::detail::Access::Complete(TimedReadyFromIo(
      WaitReactorTimed(*scheduler, socket,
                       node::ReactorInterestBits(node::ReactorInterest::Write),
                       timeout, scheduler_id, source_id, generation, epoch),
      socket, ready::Interest::Writable));
}

ready::many::Wait ready::many::wait(
    const std::span<const ready::Request> requests,
    const std::span<ready::Event> out, const std::chrono::nanoseconds timeout,
    const task::stop_token token, const ready::many::Budget budget) noexcept {
  if (timeout.count() < 0) {
    return ready::many::detail::Access::Complete(
        FailReadyMany(::rund::ReasonCode::TimerDurationInvalid));
  }
  std::uint64_t scheduler_id = 0u;
  std::uint64_t source_id = 0u;
  std::uint64_t generation = 0u;
  std::uint64_t epoch = 0u;
  if (!node::scheduler_access::StopTokenIdentity(
          token, &scheduler_id, &source_id, &generation, &epoch)) {
    return ready::many::detail::Access::Complete(
        FailReadyMany(::rund::ReasonCode::TaskInvalid));
  }
  ready::many::Result validated =
      ready::many::validation::Shape(requests, out, budget);
  if (!validated || requests.empty() || budget.max_events == 0u) {
    return ready::many::detail::Access::Complete(validated);
  }
  node::Scheduler *const scheduler = node::scheduler_access::ActiveScheduler();
  if (scheduler == nullptr) {
    return ready::many::detail::Access::Complete(
        FailReadyMany(ready::many::validation::Current(requests)
                          ? ::rund::ReasonCode::NodeRuntimeMissing
                          : ::rund::ReasonCode::IoFdInvalid));
  }
  if (scheduler->CurrentTaskIsCoroutine()) {
    return ready::many::detail::Access::Defer(
        requests, out, budget, timeout.count(), true, scheduler_id, source_id,
        generation, epoch);
  }
  return scheduler->WaitReactorMany(requests, out, timeout, budget,
                                    scheduler_id, source_id, generation, epoch);
}

} // namespace rund::net
