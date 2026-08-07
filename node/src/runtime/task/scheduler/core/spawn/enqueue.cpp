#include <rund/counter.hpp>
#include <rund/task/stats/slots.hpp>

#include "../../state/model/context.hpp"
#include "../../state/model/task.hpp"
#include "../../state/storage.hpp"

namespace rund::node {

task::Handle Scheduler::EnqueuePreparedSpawnTask(
    TaskRecord &record, const std::size_t record_index, const bool reuse_record,
    const std::uint64_t parent_id, const std::uint64_t name_hash,
    const ReadyAdmission admission) noexcept {
  if (!state_->IndexTask(record.id, record_index)) {
    DestroyTask(record);
    ReleasePreparedSpawnTaskSlot(record, record_index, reuse_record);
    return ::rund::detail::task::HandleAccess::Fail(
        ReasonCode::TaskCapacityExceeded);
  }
  if (admission == ReadyAdmission::Continuation) {
    state_->EnqueueProgress(record);
  } else if (!state_->EnqueueSpawn(record)) {
    state_->ForgetTask(record.id);
    DestroyTask(record);
    ReleasePreparedSpawnTaskSlot(record, record_index, reuse_record);
    return ::rund::detail::task::HandleAccess::Fail(
        ReasonCode::ReadyQueueCapacityExceeded);
  }

  if (reuse_record) {
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::TaskRecordReuses);
  } else {
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::TaskRecordAllocations);
  }
  if (record.coroutine_task) {
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::CoroutineTasksAdmitted);
  }
  record.resource_live = true;
  state_->resources.live_tasks.fetch_add(1u, std::memory_order_relaxed);
  ::rund::detail::counter::Accumulate(
      ::rund::detail::task::Stat(state_->evidence.metrics,
                                 ::rund::detail::task::StatSlot::Spawned),
      1u);
  RecordSpawnBatch(record.id, parent_id, record.scope_id, name_hash);

  task::Handle handle = ::rund::detail::task::HandleAccess::Make(
      record.id, state_->identity.scheduler_id, record.scope_id,
      ReasonCode::Ok);
  SchedulerThreadContext *const context = active_scheduler_context;
  if (context != nullptr && context->scheduler == this &&
      context->commit_acquired && context->split_primitive_packets) {
    FlushTaskSpawnBatch(ReasonCode::Ok);
    CompletePrimitiveCommit();
  }
  return handle;
}

} // namespace rund::node
