#include <rund/net/ready.hpp>
#include <rund/net/ready/timed.hpp>

#include <rund/task/results.hpp>

#include "../../runtime/platform/io.hpp"
#include "../../runtime/reactor/readiness/mask.hpp"
#include "../../runtime/task/scheduler/access.hpp"
#include "../../runtime/task/scheduler/state.hpp"
#include "operation.hpp"
#include "ready/ticket.hpp"
#include "registry/socket.hpp"
#include "scheduler.hpp"

namespace rund::net {
namespace {

[[nodiscard]] task::IoOp FailReady(const ::rund::ReasonCode code) noexcept {
  return ::rund::detail::task::OpAccess::io(task::Status::fail(code), 0, false,
                                            -1, 0, 0u, 0u);
}

[[nodiscard]] task::IoOp
ReadyOperation(const ::rund::detail::task::IoDecision decision,
               const bool deferred) noexcept {
  return ::rund::detail::task::OpAccess::io(decision.status, decision.revents,
                                            deferred, -1, 0, 0u, 0u);
}

[[nodiscard]] ready::Ticket
TimedReadyFromIo(const ::rund::detail::task::IoDecision result,
                 const SocketView socket,
                 const ready::Interest interest) noexcept {
  return ready::detail::Access::make(result.status.code(), socket, interest,
                                     result.revents);
}

[[nodiscard]] ready::Wait WaitSocket(const SocketView socket,
                                     const ready::Interest public_interest,
                                     const short interest) noexcept {
  node::Scheduler *const scheduler = node::scheduler_access::ActiveScheduler();
  if (scheduler == nullptr) {
    const ::rund::ReasonCode code = IsCurrentSocket(socket)
                                        ? ::rund::ReasonCode::NodeRuntimeMissing
                                        : ::rund::ReasonCode::IoFdInvalid;
    return ready::detail::Access::wait(FailReady(code), socket, public_interest,
                                       interest, false);
  }
  if (scheduler->CurrentTaskIsCoroutine()) {
    return ready::detail::Access::wait(
        ReadyOperation(
            ::rund::detail::task::IoDecision{.status = task::Status::success()},
            true),
        socket, public_interest, interest, true);
  }
  const ::rund::detail::task::IoDecision result =
      WaitReactor(*scheduler, socket, interest);
  return ready::detail::Access::wait(ReadyOperation(result, false), socket,
                                     public_interest, interest, false);
}

[[nodiscard]] ready::timed::Wait
WaitSocketTimed(const SocketView socket, const ready::Interest public_interest,
                const short interest,
                const std::chrono::nanoseconds timeout) noexcept {
  if (timeout.count() < 0) {
    return ready::timed::detail::Access::Complete(ready::detail::Access::make(
        ::rund::ReasonCode::TimerDurationInvalid, socket, public_interest));
  }
  node::Scheduler *const scheduler = node::scheduler_access::ActiveScheduler();
  if (scheduler == nullptr) {
    const ::rund::ReasonCode code = IsCurrentSocket(socket)
                                        ? ::rund::ReasonCode::NodeRuntimeMissing
                                        : ::rund::ReasonCode::IoFdInvalid;
    return ready::timed::detail::Access::Complete(
        ready::detail::Access::make(code, socket, public_interest));
  }
  if (scheduler->CurrentTaskIsCoroutine()) {
    return ready::timed::detail::Access::Defer(socket, public_interest,
                                               interest, timeout.count());
  }
  return ready::timed::detail::Access::Complete(
      TimedReadyFromIo(WaitReactorTimed(*scheduler, socket, interest, timeout),
                       socket, public_interest));
}

} // namespace

ready::Ticket ready::Wait::wait() && noexcept {
  if (deferred_) {
    if (!IsCurrentSocket(socket_)) {
      return ready::detail::Access::make(::rund::ReasonCode::IoFdInvalid,
                                         socket_, interest_);
    }
    return ready::detail::Access::make(::rund::ReasonCode::TaskContextMissing,
                                       socket_, interest_);
  }
  return ready::detail::Access::make(operation_.code(), socket_, interest_,
                                     operation_.revents());
}

ready::Wait ready::detail::fail(const SocketView socket,
                                const ready::Interest interest,
                                const ::rund::ReasonCode code) noexcept {
  return ready::detail::Access::wait(FailReady(code), socket, interest, 0,
                                     false);
}

ready::Wait ready::detail::complete(const SocketView socket,
                                    const ready::Interest interest) noexcept {
  node::Scheduler *const scheduler = node::scheduler_access::ActiveScheduler();
  if (scheduler == nullptr) {
    const ::rund::ReasonCode code = IsCurrentSocket(socket)
                                        ? ::rund::ReasonCode::NodeRuntimeMissing
                                        : ::rund::ReasonCode::IoFdInvalid;
    return fail(socket, interest, code);
  }
  if (!scheduler->CurrentTaskIsCoroutine()) {
    return fail(socket, interest, ::rund::ReasonCode::TaskContextMissing);
  }
  return ready::detail::Access::wait(ReadyOperation(
                                         ::rund::detail::task::IoDecision{
                                             .status = task::Status::success(),
                                         },
                                         false),
                                     socket, interest, 0, false);
}

bool ready::Awaiter::await_suspend(
    const std::coroutine_handle<> handle) noexcept {
  if (!deferred_) {
    return operation_.await_suspend(handle);
  }
  decision_ = ::rund::detail::task::AwaitAccess::SuspendCoroutineNetIo(
      socket_, native_interest_);
  return decision_.status && decision_.suspend;
}

ready::Ticket ready::Awaiter::await_resume() noexcept {
  const task::IoResult result =
      deferred_
          ? ::rund::detail::task::AwaitAccess::CompleteCoroutineNetIo(decision_)
          : operation_.await_resume();
  return ready::detail::Access::make(result.code(), socket_, interest_,
                                     result.revents());
}

ready::Ticket ready::timed::Wait::wait() && noexcept {
  if (deferred_ || suspended_) {
    if (!IsCurrentSocket(socket_)) {
      return ready::detail::Access::make(::rund::ReasonCode::IoFdInvalid,
                                         socket_, public_interest_);
    }
    return ready::detail::Access::make(::rund::ReasonCode::TaskContextMissing,
                                       socket_, public_interest_);
  }
  return std::move(result_);
}

ready::Wait ready::read(const SocketView socket) noexcept {
  return WaitSocket(socket, ready::Interest::Readable,
                    node::ReactorInterestBits(node::ReactorInterest::Read));
}

ready::Wait ready::write(const SocketView socket) noexcept {
  return WaitSocket(socket, ready::Interest::Writable,
                    node::ReactorInterestBits(node::ReactorInterest::Write));
}

ready::timed::Wait
ready::timed::read(const SocketView socket,
                   const std::chrono::nanoseconds timeout) noexcept {
  return WaitSocketTimed(socket, ready::Interest::Readable,
                         node::ReactorInterestBits(node::ReactorInterest::Read),
                         timeout);
}

ready::timed::Wait
ready::timed::write(const SocketView socket,
                    const std::chrono::nanoseconds timeout) noexcept {
  return WaitSocketTimed(
      socket, ready::Interest::Writable,
      node::ReactorInterestBits(node::ReactorInterest::Write), timeout);
}

} // namespace rund::net
