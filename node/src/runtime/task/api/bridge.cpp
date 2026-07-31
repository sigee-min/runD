#include "../../platform/io.hpp"
#include "../scheduler/access.hpp"
#include "../scheduler/state.hpp"

#include <rund/session/memory.hpp>
#include <rund/task/api/access.hpp>
#include <rund/task/await/access.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <utility>

namespace rund::detail::task {

::rund::task::Handle ApiAccess::SpawnRunnable(const char *const name,
                                              Callable callable) noexcept {
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr) {
    return HandleAccess::Fail(ReasonCode::NodeRuntimeMissing);
  }
  return scheduler->Spawn(name, std::move(callable));
}

::rund::task::Handle
ApiAccess::SpawnCoroutine(const char *const name,
                          ::rund::task::Task<void> &&task) noexcept {
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr) {
    CoroutineStart coroutine = TakeCoroutine(std::move(task));
    DestroyCoroutine(coroutine);
    return HandleAccess::Fail(ReasonCode::NodeRuntimeMissing);
  }
  return scheduler->Spawn(name, std::move(task));
}

Spawned ApiAccess::SpawnAwaited(CoroutineStart start) noexcept {
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr) {
    DestroyCoroutine(start);
    return Spawned{.task = HandleAccess::Fail(ReasonCode::NodeRuntimeMissing)};
  }
  return scheduler->SpawnAwaited(start);
}

::rund::task::Status
ApiAccess::JoinMany(const ::rund::task::Handle *const handles,
                    const std::size_t count) noexcept {
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr) {
    return ::rund::task::Status::fail(ReasonCode::NodeRuntimeMissing);
  }
  return scheduler->Join(handles, count);
}

::rund::task::Status ApiAccess::EnterScope(Callable callable) noexcept {
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr) {
    return ::rund::task::Status::fail(ReasonCode::NodeRuntimeMissing);
  }
  const node::ScopeToken token = scheduler->BeginScope();
  if (token.code != ReasonCode::Ok) {
    return ::rund::task::Status::fail(token.code);
  }
  try {
    callable();
  } catch (...) {
    (void)scheduler->EndScope(token);
    return ::rund::task::Status::fail(ReasonCode::TaskScopeCallbackFailed);
  }
  return scheduler->EndScope(token);
}

} // namespace rund::detail::task

namespace rund::task {

YieldOp yield() noexcept {
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr) {
    return ::rund::detail::task::OpAccess::yield(
        Status::fail(ReasonCode::NodeRuntimeMissing), false);
  }
  if (scheduler->CurrentTaskIsCoroutine()) {
    return ::rund::detail::task::OpAccess::yield(Status::success(), true);
  }
  const ::rund::detail::task::AwaitDecision decision = scheduler->Yield();
  return ::rund::detail::task::OpAccess::yield(decision.status, false);
}

SleepOp sleep(const std::chrono::nanoseconds duration) noexcept {
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr) {
    return ::rund::detail::task::OpAccess::sleep(
        Status::fail(ReasonCode::NodeRuntimeMissing), duration, false);
  }
  if (scheduler->CurrentTaskIsCoroutine()) {
    return ::rund::detail::task::OpAccess::sleep(Status::success(), duration,
                                                 true);
  }
  const ::rund::detail::task::AwaitDecision decision =
      scheduler->Sleep(duration);
  return ::rund::detail::task::OpAccess::sleep(decision.status, duration,
                                               false);
}

} // namespace rund::task
