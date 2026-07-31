#include <rund/task/await/access.hpp>
#include <optional>
#include <rund/net/ready/many.hpp>

#include "../../../runtime/task/scheduler/access.hpp"
#include "../../../runtime/task/scheduler/state.hpp"
#include "../interest.hpp"
#include "../operation.hpp"
#include "validation.hpp"

namespace rund::net {

ready::many::Result ready::many::Wait::wait() const noexcept {
  if (!deferred_ && !suspended_) {
    return result_;
  }
  return ready::many::validation::Current(requests_)
             ? Result{::rund::ReasonCode::TaskContextMissing}
             : Result{::rund::ReasonCode::IoFdInvalid};
}

bool ready::many::Awaiter::await_suspend(std::coroutine_handle<>) noexcept {
  if (operation_.deferred_) {
    operation_ = rund::detail::task::AwaitAccess::SuspendCoroutineReadyMany(
        std::move(operation_));
  }
  return operation_.suspended_;
}

ready::many::Result ready::many::Awaiter::await_resume() noexcept {
  return rund::detail::task::AwaitAccess::CompleteCoroutineReadyMany(
      std::move(operation_));
}

ready::many::Wait
ready::many::wait(const std::span<const ready::Request> requests,
                  const std::span<ready::Event> out,
                  const ready::many::Budget budget) noexcept {
  ready::many::Result validated =
      ready::many::validation::Shape(requests, out, budget);
  if (!validated || requests.empty() || budget.max_events == 0u) {
    return ready::many::detail::Access::Complete(validated);
  }
  node::Scheduler *const scheduler = node::scheduler_access::ActiveScheduler();
  if (scheduler == nullptr) {
    return ready::many::detail::Access::Complete(ready::many::validation::Fail(
        ready::many::validation::Current(requests)
            ? ::rund::ReasonCode::NodeRuntimeMissing
            : ::rund::ReasonCode::IoFdInvalid));
  }
  if (scheduler->CurrentTaskIsCoroutine()) {
    return ready::many::detail::Access::Defer(requests, out, budget);
  }
  return scheduler->WaitReactorMany(requests, out, std::nullopt, budget);
}

ready::many::Wait
ready::many::wait(const std::span<const ready::Request> requests,
                  const std::span<ready::Event> out,
                  const std::chrono::nanoseconds timeout,
                  const ready::many::Budget budget) noexcept {
  if (timeout.count() < 0) {
    return ready::many::detail::Access::Complete(ready::many::validation::Fail(
        ::rund::ReasonCode::TimerDurationInvalid));
  }
  ready::many::Result validated =
      ready::many::validation::Shape(requests, out, budget);
  if (!validated || requests.empty() || budget.max_events == 0u) {
    return ready::many::detail::Access::Complete(validated);
  }
  node::Scheduler *const scheduler = node::scheduler_access::ActiveScheduler();
  if (scheduler == nullptr) {
    return ready::many::detail::Access::Complete(ready::many::validation::Fail(
        ready::many::validation::Current(requests)
            ? ::rund::ReasonCode::NodeRuntimeMissing
            : ::rund::ReasonCode::IoFdInvalid));
  }
  if (scheduler->CurrentTaskIsCoroutine()) {
    return ready::many::detail::Access::Defer(requests, out, budget,
                                              timeout.count(), true);
  }
  return scheduler->WaitReactorMany(requests, out, timeout, budget);
}

} // namespace rund::net
