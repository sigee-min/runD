#include "../state/storage.hpp"
#include "../state/task/commit.hpp"

#include <rund/task/api/access.hpp>

namespace rund::node {

task::Handle
Scheduler::Spawn(const char *const name,
                 ::rund::detail::task::Callable callable) noexcept {
  if (!callable) {
    return ::rund::detail::task::HandleAccess::Fail(ReasonCode::TaskInvalid);
  }
  return SpawnPrepared(name, &callable, {}, nullptr, ReadyAdmission::Spawn);
}

task::Handle Scheduler::Spawn(const char *const name,
                              task::Task<void> &&task) noexcept {
  ::rund::detail::task::CoroutineStart coroutine =
      ::rund::detail::task::TakeCoroutine(std::move(task));
  if (coroutine.frame == nullptr) {
    return ::rund::detail::task::HandleAccess::Fail(ReasonCode::TaskInvalid);
  }
  return SpawnPrepared(name, nullptr, coroutine, nullptr,
                       ReadyAdmission::Spawn);
}

::rund::detail::task::Spawned
Scheduler::SpawnWithCompletion(const char *const name,
                               const ::rund::detail::task::CoroutineStart start,
                               const ReadyAdmission admission) noexcept {
  ::rund::detail::task::ResultRef observer{};
  const task::Handle task =
      SpawnPrepared(name, nullptr, start, &observer, admission);
  return ::rund::detail::task::Spawned{.task = task, .result = observer};
}

::rund::detail::task::Spawned Scheduler::SpawnObserved(
    const char *const name,
    const ::rund::detail::task::CoroutineStart start) noexcept {
  return SpawnWithCompletion(name, start, ReadyAdmission::Spawn);
}

::rund::detail::task::Spawned Scheduler::SpawnAwaited(
    const ::rund::detail::task::CoroutineStart start) noexcept {
  return SpawnWithCompletion("nested-task", start,
                             ReadyAdmission::Continuation);
}

task::Handle
Scheduler::SpawnPrepared(const char *const name,
                         ::rund::detail::task::Callable *const callable,
                         const ::rund::detail::task::CoroutineStart coroutine,
                         ::rund::detail::task::ResultRef *const observer,
                         const ReadyAdmission admission) noexcept {
  if (callable == nullptr && coroutine.frame == nullptr) {
    if (observer != nullptr) {
      *observer = {};
    }
    return ::rund::detail::task::HandleAccess::Fail(
        coroutine.code == ReasonCode::Ok ? ReasonCode::TaskInvalid
                                         : coroutine.code);
  }
  ControlCommitScope commit{*this};
  return SpawnPreparedCommitted(name, callable, coroutine, observer, admission);
}

} // namespace rund::node
