#include <rund/task/stats/slots.hpp>

#include "../state/model/task.hpp"
#include "../state/storage.hpp"
#include "../state/storage/check.hpp"

#include <rund/task/coroutine.hpp>

namespace rund::node {
namespace {

void ReleaseCompletion(TaskRecord &record, CompletionPool &pool) noexcept {
  const CompletionLease completion = pool.lease(record.completion);
  if (!completion) {
    return;
  }
  (void)CompletionPool::terminate(completion, ReasonCode::TaskFailed);
  CompletionPool::release(completion);
  record.completion = {};
}

} // namespace

void Scheduler::DestroyTask(TaskRecord &record) noexcept {
  if (record.resource_live) {
    state_->resources.live_tasks.fetch_sub(1u, std::memory_order_relaxed);
    record.resource_live = false;
  }
  if (!record.coroutine_task && record.callable) {
    state_->resources.callable_pool.release(record.callable);
    record.callable = nullptr;
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::CallableResets);
  }
  ReleaseCompletion(record, state_->resources.completion_pool);
  if (record.coroutine_task && record.coroutine_frame != nullptr &&
      record.coroutine_ops != nullptr &&
      record.coroutine_ops->destroy != nullptr) {
    record.coroutine_ops->destroy(record.coroutine_frame);
    record.coroutine_frame = nullptr;
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::CoroutineFrameDestroys);
  }
  record.coroutine_task = false;
  record.coroutine_ops = nullptr;
  record.coroutine_parked = false;
  record.wait_id = 0u;
  record.wait_source_id = 0u;
  record.wait_token = 0u;
  record.lane_segment_side_exit = false;
  record.wait_result = ReasonCode::Ok;
  record.io_revents = 0;
  record.io_result = ReasonCode::Ok;
  record.wake_next = nullptr;
  record.wake_ticket = 0u;
}

void Scheduler::DestroyLaneCallable(TaskRecord &record) noexcept {
  if (!record.coroutine_task) {
    state_->resources.callable_pool.destroy(record.callable);
  }
}

void Scheduler::RetireTaskRecord(TaskRecord &record,
                                 const bool count_record_retire) noexcept {
  state_->RequireSequencer();
  if (record.recyclable || record.id == 0u) {
    return;
  }
  const std::uint64_t old_id = record.id;
  const std::size_t index = state_->IndexFor(old_id);
  DestroyTask(record);
  state_->ForgetTask(old_id);
  record.id = 0u;
  record.scope_id = 1u;
  record.dynamic_scope_id = 1u;
  record.lane_segment_side_exit = false;
  record.coroutine_parked = false;
  record.state = TaskState::Completed;
  record.failure_code = ReasonCode::Ok;
  record.recyclable = true;
  if (index < state_->ready.records.size()) {
    record.wake_ticket = state_->ready.free_record_head;
    state_->ready.free_record_head = static_cast<std::uint32_t>(index + 1u);
    if (count_record_retire) {
      ++::rund::detail::task::Stat(
          state_->evidence.metrics,
          ::rund::detail::task::StatSlot::TaskRecordRetires);
    }
  }
}

void Scheduler::RetireTask(TaskRecord &record) noexcept {
  RetireTaskRecord(record, true);
}

void Scheduler::QueueRootJoinRetireKnown(TaskRecord &record,
                                         const std::uint64_t task_id,
                                         const ReasonCode code) noexcept {
  state_->RequireSequencer();
  auto &range = state_->batches.pending_root_join_range;
  const bool can_extend =
      range.active && range.code == code &&
      range.last_task_id != std::numeric_limits<std::uint64_t>::max() &&
      range.last_task_id + 1u == task_id;
  if (!can_extend && range.active) {
    FlushPendingRootJoinRetireBatch();
  }
  if (can_extend) {
    range.last_task_id = task_id;
    ++range.logical_tasks;
  } else {
    range = SchedulerBatchState::PendingRootJoinRange{
        .active = true,
        .first_task_id = task_id,
        .last_task_id = task_id,
        .logical_tasks = 1u,
        .code = code,
    };
  }
  RetireTaskRecord(record, false);
}

void Scheduler::FlushPendingRootJoinRetireBatch() noexcept {
  state_->RequireSequencer();
  auto &range = state_->batches.pending_root_join_range;
  if (!range.active || range.logical_tasks == 0u) {
    range = SchedulerBatchState::PendingRootJoinRange{};
    return;
  }
  FlushTaskSpawnBatch(ReasonCode::Ok);
  FlushYieldBatch(ReasonCode::Ok);
  FlushPendingRootSubmit();
  RecordJoinRetireBatch(range.first_task_id, range.last_task_id, range.code,
                        range.logical_tasks);
  range = SchedulerBatchState::PendingRootJoinRange{};
}

} // namespace rund::node
