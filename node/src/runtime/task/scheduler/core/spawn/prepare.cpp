#include <rund/task/stats/slots.hpp>

#include "../../state/model/task.hpp"
#include "../../state/storage.hpp"
#include "reject.hpp"

#include <rund/host/hash.hpp>
#include <rund/task/coroutine.hpp>

#include <string_view>

namespace rund::node {

task::Handle Scheduler::SpawnPreparedCommitted(
    const char *const name, ::rund::detail::task::Callable *const callable,
    const ::rund::detail::task::CoroutineStart coroutine,
    ::rund::detail::task::ResultRef *const observer,
    const ReadyAdmission admission) noexcept {
  if (name == nullptr || name[0] == '\0') {
    if (observer != nullptr) {
      *observer = {};
    }
    DestroyRejectedSpawnPayload(coroutine);
    return ::rund::detail::task::HandleAccess::Fail(ReasonCode::TaskInvalid);
  }
  const std::string_view task_name{name};
  const std::uint64_t name_hash =
      ::rund::host::hash_string(task_name.data(), task_name.size()).value;
  (void)TrapLaneOwnedSegmentPrimitive(
      ::rund::detail::task::OperationKind::Spawn);
  EnsureCurrentCommit();
  const bool root_spawn = CurrentTaskId() == 0u;
  const ReasonCode budget = ValidateSpawnBudget(admission);
  if (budget != ReasonCode::Ok) {
    DestroyRejectedSpawnPayload(coroutine);
    return ::rund::detail::task::HandleAccess::Fail(budget);
  }

  bool reuse_record = false;
  const std::size_t record_index = ClaimSpawnTaskSlot(reuse_record);
  TaskRecord &record = state_->ready.records[record_index];
  CompletionLease completion{};
  if (coroutine.frame != nullptr && observer != nullptr) {
    completion = state_->resources.completion_pool.claim();
    if (!completion) {
      ReleasePreparedSpawnTaskSlot(record, record_index, reuse_record);
      DestroyRejectedSpawnPayload(coroutine);
      return ::rund::detail::task::HandleAccess::Fail(
          ReasonCode::TaskCompletionCapacity);
    }
    *observer = state_->resources.completion_pool.observe_ref(completion);
  }
  const std::uint64_t parent_task_id = CurrentTaskId();
  const std::uint64_t scope_id = CurrentScopeId();
  if (!MaterializePreparedSpawnTask(record, callable, coroutine, completion,
                                    parent_task_id, scope_id, root_spawn)) {
    ReleasePreparedSpawnTaskSlot(record, record_index, reuse_record);
    if (coroutine.frame != nullptr && coroutine.ops != nullptr &&
        coroutine.ops->destroy != nullptr) {
      coroutine.ops->destroy(coroutine.frame);
      ++::rund::detail::task::Stat(
          state_->evidence.metrics,
          ::rund::detail::task::StatSlot::CoroutineFrameDestroys);
    }
    RejectSpawnCompletion(completion, ReasonCode::TaskCapacityExceeded);
    return ::rund::detail::task::HandleAccess::Fail(
        ReasonCode::TaskCapacityExceeded);
  }
  if (completion &&
      !CompletionPool::transition(completion, task::Phase::Ready)) {
    DestroyTask(record);
    ReleasePreparedSpawnTaskSlot(record, record_index, reuse_record);
    return ::rund::detail::task::HandleAccess::Fail(
        ReasonCode::TaskStateTransitionInvalid);
  }
  return EnqueuePreparedSpawnTask(record, record_index, reuse_record,
                                  parent_task_id, name_hash, admission);
}

} // namespace rund::node
