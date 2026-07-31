#include <rund/task/stats/slots.hpp>

#include "../../../state/storage/check.hpp"
#include "local.hpp"

namespace rund::node {

void Scheduler::FlushYieldBatch(const ReasonCode side_exit_code) noexcept {
  state_->RequireSequencer();
  auto &batch = state_->batches.yield_resume_batch;
  if (!batch.active || batch.logical_count == 0u) {
    batch = SchedulerBatchState::YieldResumeBatch{};
    return;
  }
  const std::uint64_t order_hash =
      YieldBatchOrderHash(true, state_->plan.task(batch.task_id),
                          batch.logical_count, batch.logical_yields);
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::YieldBatchPackets);
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::YieldBatchLogicalEvents) +=
      batch.logical_count;
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::YieldBatchOrderHashFastPaths);
  RecordPhysical(::rund::detail::task::OperationKind::YieldBatch,
                 ReasonCode::Ok, batch.task_id, batch.task_id, 0u, 0u, -1, 0, 0,
                 0, batch.logical_count, 0u, 0u, batch.logical_yields, 0u, 0u,
                 0u, batch.logical_count, order_hash, side_exit_code);
  batch = SchedulerBatchState::YieldResumeBatch{};
}

void Scheduler::RecordYieldBatch(const std::uint64_t task_id) noexcept {
  state_->RequireSequencer();
  FlushRootSingleJoinEpoch(ReasonCode::Ok);
  FlushTaskSpawnBatch(ReasonCode::Ok);
  const bool consumed_root_submit = ConsumePendingRootSubmit(task_id);
  if (!consumed_root_submit) {
    FlushYieldBatch(ReasonCode::Ok);
    RecordPhysical(::rund::detail::task::OperationKind::RootSubmit,
                   ReasonCode::Ok, task_id);
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::SchedulerResumeRootSubmitFlushed);
    const std::uint64_t logical_count = 1u;
    const std::uint64_t order_hash =
        YieldBatchOrderHash(false, state_->plan.task(task_id), logical_count);
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::YieldBatchPackets);
    ::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::YieldBatchLogicalEvents) +=
        logical_count;
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::YieldBatchOrderHashFastPaths);
    RecordPhysical(::rund::detail::task::OperationKind::YieldBatch,
                   ReasonCode::Ok, task_id, task_id, 0u, 0u, -1, 0, 0, 0,
                   logical_count, 0u, 0u, 1u, 0u, 0u, 0u, logical_count,
                   order_hash, ReasonCode::Ok);
    return;
  }
  auto &batch = state_->batches.yield_resume_batch;
  if (!batch.active || batch.task_id != task_id) {
    FlushYieldBatch(ReasonCode::Ok);
    batch.active = true;
    batch.task_id = task_id;
  }
  ++batch.logical_yields;
  batch.logical_count += 2u;
}

void Scheduler::RecordJoinBatch(const std::uint64_t target_task_id,
                                const ReasonCode code) noexcept {
  state_->RequireSequencer();
  FlushRootSingleJoinEpoch(ReasonCode::Ok);
  FlushTaskSpawnBatch(ReasonCode::Ok);
  FlushYieldBatch(ReasonCode::Ok);
  FlushPendingRootSubmit();
  constexpr std::uint64_t logical_count = 2u;
  const std::uint64_t task_id = CurrentTaskId();
  const std::uint64_t order_hash = JoinBatchOrderHash(
      state_->plan.task(task_id), state_->plan.task(target_task_id), code,
      logical_count);
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::JoinBatchPackets);
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::JoinBatchLogicalEvents) += logical_count;
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::JoinBatchOrderHashFastPaths);
  RecordPhysical(
      ::rund::detail::task::OperationKind::JoinBatch, code, task_id,
      target_task_id, 0u, 0u, -1, 0, 0, 0, target_task_id, 0u, 0u,
      static_cast<std::uint64_t>(::rund::detail::task::OperationKind::JoinPark),
      static_cast<std::uint64_t>(::rund::detail::task::OperationKind::JoinWake),
      0u, 0u, logical_count, order_hash, ReasonCode::Ok);
}

void Scheduler::RecordJoinRetireBatch(
    const std::uint64_t first_task_id, const std::uint64_t last_task_id,
    const ReasonCode code, const std::uint64_t logical_tasks) noexcept {
  state_->RequireSequencer();
  if (logical_tasks == 0u) {
    return;
  }
  constexpr std::uint64_t logical_per_task = 2u;
  FlushRootSingleJoinEpoch(ReasonCode::Ok);
  const std::uint64_t logical_count = logical_tasks * logical_per_task;
  const std::uint64_t task_id = CurrentTaskId();
  std::uint64_t order_hash =
      JoinBatchOrderHash(state_->plan.task(task_id),
                         state_->plan.task(first_task_id), code, logical_count);
  MixHash(order_hash, state_->plan.task(last_task_id));
  MixHash(order_hash, logical_tasks);
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::JoinBatchPackets);
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::JoinBatchLogicalEvents) += logical_count;
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::JoinBatchOrderHashFastPaths);
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::JoinOwnerLaneTerminalLogPackets);
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::JoinOwnerLaneTerminalLogicalTasks) +=
      logical_tasks;
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::RootJoinRetireBatchPackets);
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::RootJoinRetireBatchLogicalTasks) +=
      logical_tasks;
  if (logical_tasks > 1u) {
    ::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::JoinSlotValidations) += logical_tasks;
    ::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::JoinSlotTerminalLookups) +=
        logical_tasks;
    ::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::JoinSlotRetireLookups) += logical_tasks;
    ::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::JoinSlotIdIndexLookupsAvoided) +=
        logical_tasks * 3u;
  }
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::TaskRecordRetires) += logical_tasks;
  if (logical_tasks > 1u) {
    ::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::RootPerTaskJoinWakesAvoided) +=
        logical_tasks - 1u;
    ::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::RootPerTaskRetireCallsAvoided) +=
        logical_tasks - 1u;
  }
  RecordPhysical(
      ::rund::detail::task::OperationKind::JoinBatch, code, task_id,
      first_task_id, 0u, 0u, -1, 0, 0, 0, last_task_id, 0u, 0u,
      static_cast<std::uint64_t>(::rund::detail::task::OperationKind::JoinPark),
      static_cast<std::uint64_t>(::rund::detail::task::OperationKind::JoinWake),
      0u, 0u, logical_count, order_hash, ReasonCode::Ok);
}

} // namespace rund::node
