#include <rund/task/stats/slots.hpp>

#include "../../../state/storage/check.hpp"
#include "local.hpp"

namespace rund::node {

void Scheduler::FlushRootSingleJoinEpoch(
    const ReasonCode side_exit_code) noexcept {
  state_->RequireSequencer();
  auto &epoch = state_->batches.root_single_join_epoch;
  if (!epoch.active || epoch.logical_tasks == 0u) {
    epoch = SchedulerBatchState::RootSingleJoinEpoch{};
    return;
  }
  const std::uint64_t order_hash = RootSingleJoinEpochOrderHash(
      state_->plan.task(epoch.first_task_id),
      state_->plan.task(epoch.last_task_id),
      state_->plan.task(epoch.parent_task_id),
      state_->plan.scope(epoch.scope_id), epoch.logical_tasks,
      epoch.logical_events, epoch.spawn_participant_hash);
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::RootSingleJoinEpochPackets);
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::RootSingleJoinEpochLogicalTasks) +=
      epoch.logical_tasks;
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::RootSingleJoinEpochLogicalEvents) +=
      epoch.logical_events;
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::RootSingleJoinEpochOrderHashFastPaths);
  RecordPhysical(
      ::rund::detail::task::OperationKind::TaskRootJoinEpochBatch,
      ReasonCode::Ok, epoch.first_task_id, epoch.last_task_id, 0u, 0u, -1, 0, 0,
      0, epoch.logical_events, 0u, 0u, epoch.parent_task_id,
      static_cast<std::uint64_t>(
          ::rund::detail::task::OperationKind::TaskSpawnBatch),
      static_cast<std::uint64_t>(
          ::rund::detail::task::OperationKind::TaskTerminalBatch),
      epoch.epoch_id, epoch.logical_tasks, order_hash, side_exit_code);
  epoch = SchedulerBatchState::RootSingleJoinEpoch{};
}

bool Scheduler::RecordRootSingleJoinEpochTask(
    const ::rund::detail::task::OperationKind terminal_kind,
    const ReasonCode code, const std::uint64_t task_id) noexcept {
  state_->RequireSequencer();
  SchedulerThreadContext *const context = active_scheduler_context;
  if (context == nullptr || context->scheduler != this ||
      !context->root_exclusive_commit ||
      terminal_kind != ::rund::detail::task::OperationKind::Complete ||
      code != ReasonCode::Ok) {
    return false;
  }
  auto &spawn = state_->batches.task_spawn_batch;
  if (!spawn.active || spawn.logical_count != 1u ||
      spawn.first_task_id != task_id || spawn.last_task_id != task_id) {
    return false;
  }
  if (!ConsumePendingRootSubmit(task_id)) {
    return false;
  }
  auto &epoch = state_->batches.root_single_join_epoch;
  if (epoch.active && (epoch.parent_task_id != spawn.parent_task_id ||
                       epoch.scope_id != spawn.scope_id ||
                       epoch.last_task_id + 1u != task_id)) {
    FlushRootSingleJoinEpoch(ReasonCode::Ok);
  }
  if (!epoch.active) {
    epoch.active = true;
    epoch.epoch_id = state_->identity.next_trace_epoch_id++;
    epoch.first_task_id = task_id;
    epoch.parent_task_id = spawn.parent_task_id;
    epoch.scope_id = spawn.scope_id;
  }
  epoch.last_task_id = task_id;
  ++epoch.logical_tasks;
  epoch.logical_events += 3u;
  MixHash(epoch.spawn_participant_hash, spawn.order_hash);
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::RootSingleJoinSpawnPacketsElided);
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::RootSingleJoinTerminalPacketsElided);
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::RootSingleJoinRootSubmitPacketsElided);
  spawn = SchedulerBatchState::TaskSpawnBatch{};
  const std::uint64_t flush_tasks = std::max<std::uint64_t>(
      1u,
      std::max<std::uint64_t>(state_->resources.limits.task_capacity,
                              state_->resources.limits.ready_queue_capacity));
  if (epoch.logical_tasks >= flush_tasks) {
    FlushRootSingleJoinEpoch(ReasonCode::Ok);
  }
  return true;
}

void Scheduler::FlushTaskSpawnBatch(const ReasonCode side_exit_code) noexcept {
  state_->RequireSequencer();
  if (state_->batches.root_single_join_epoch.active) {
    FlushRootSingleJoinEpoch(ReasonCode::Ok);
  }
  auto &batch = state_->batches.task_spawn_batch;
  if (!batch.active || batch.logical_count == 0u) {
    batch = SchedulerBatchState::TaskSpawnBatch{};
    return;
  }
  const std::uint64_t order_hash =
      SpawnBatchOrderHash(state_->plan.task(batch.first_task_id),
                          state_->plan.task(batch.last_task_id),
                          state_->plan.task(batch.parent_task_id),
                          state_->plan.scope(batch.scope_id),
                          batch.logical_count, batch.order_hash);
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::TaskSpawnBatchPackets);
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::TaskSpawnBatchLogicalEvents) +=
      batch.logical_count;
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::TaskSpawnBatchOrderHashFastPaths);
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::TaskSpawnBatchOrderHashElidedEvents) +=
      batch.logical_count;
  RecordPhysical(::rund::detail::task::OperationKind::TaskSpawnBatch,
                 ReasonCode::Ok, batch.first_task_id, batch.last_task_id, 0u,
                 0u, -1, 0, 0, 0, batch.logical_count, 0u, 0u,
                 batch.parent_task_id, 0u, batch.scope_id, 0u,
                 batch.logical_count, order_hash, side_exit_code);
  batch = SchedulerBatchState::TaskSpawnBatch{};
}

void Scheduler::RecordSpawnBatch(const std::uint64_t task_id,
                                 const std::uint64_t parent_task_id,
                                 const std::uint64_t scope_id,
                                 const std::uint64_t name_hash) noexcept {
  state_->RequireSequencer();
  FlushPendingRootSubmit();
  auto &batch = state_->batches.task_spawn_batch;
  if (!batch.active) {
    batch.active = true;
    batch.first_task_id = task_id;
    batch.parent_task_id = parent_task_id;
    batch.scope_id = scope_id;
  } else if (batch.parent_task_id != parent_task_id ||
             batch.scope_id != scope_id ||
             (batch.logical_count != 0u &&
              batch.last_task_id + 1u != task_id)) {
    FlushTaskSpawnBatch(ReasonCode::Ok);
    batch.active = true;
    batch.first_task_id = task_id;
    batch.parent_task_id = parent_task_id;
    batch.scope_id = scope_id;
  }
  batch.last_task_id = task_id;
  ++batch.logical_count;
  batch.order_hash =
      AppendSpawnParticipantHash(batch.order_hash, state_->plan.task(task_id),
                                 state_->plan.task(parent_task_id),
                                 state_->plan.scope(scope_id), name_hash);
}

void Scheduler::RecordTerminalBatch(
    const ::rund::detail::task::OperationKind terminal_kind,
    const ReasonCode code, const std::uint64_t task_id) noexcept {
  state_->RequireSequencer();
  if (RecordRootSingleJoinEpochTask(terminal_kind, code, task_id)) {
    return;
  }
  FlushTaskSpawnBatch(ReasonCode::Ok);
  FlushYieldBatch(ReasonCode::Ok);
  const bool consumed_root_submit = ConsumePendingRootSubmit(task_id);
  if (!consumed_root_submit) {
    RecordPhysical(::rund::detail::task::OperationKind::RootSubmit,
                   ReasonCode::Ok, task_id);
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::SchedulerResumeRootSubmitFlushed);
  }
  const std::uint64_t logical_count = consumed_root_submit ? 2u : 1u;
  const std::uint64_t order_hash =
      TerminalBatchOrderHash(consumed_root_submit, terminal_kind, code,
                             state_->plan.task(task_id), logical_count);
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::TaskTerminalBatchPackets);
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::TaskTerminalBatchLogicalEvents) +=
      logical_count;
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::TaskTerminalBatchOrderHashFastPaths);
  RecordPhysical(::rund::detail::task::OperationKind::TaskTerminalBatch, code,
                 task_id, task_id, 0u, 0u, -1, 0, 0, 0, logical_count, 0u, 0u,
                 static_cast<std::uint64_t>(terminal_kind), 0u, 0u, 0u,
                 logical_count, order_hash, ReasonCode::Ok);
}

void Scheduler::RecordTerminalRangeBatch(
    const ::rund::detail::task::OperationKind terminal_kind,
    const ReasonCode code, const std::uint64_t first_task_id,
    const std::uint64_t last_task_id, const std::uint64_t first_ticket,
    const std::uint64_t last_ticket, const std::uint64_t logical_tasks,
    const std::uint64_t order_hash, const bool includes_root_submit) noexcept {
  state_->RequireSequencer();
  if (logical_tasks == 0u) {
    return;
  }
  FlushRootSingleJoinEpoch(ReasonCode::Ok);
  FlushTaskSpawnBatch(ReasonCode::Ok);
  FlushYieldBatch(ReasonCode::Ok);
  FlushPendingRootSubmit();
  const std::uint64_t logical_count =
      logical_tasks * (includes_root_submit ? 2u : 1u);
  const std::uint64_t canonical_hash = TerminalRangeBatchOrderHash(
      includes_root_submit, terminal_kind, code,
      state_->plan.task(first_task_id), state_->plan.task(last_task_id),
      state_->plan.ticket(first_ticket), state_->plan.ticket(last_ticket),
      logical_tasks, order_hash);
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::TaskTerminalBatchPackets);
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::TaskTerminalBatchLogicalEvents) +=
      logical_count;
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::TaskTerminalBatchOrderHashFastPaths);
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::DeterministicCommitRingMerges);
  RecordPhysical(::rund::detail::task::OperationKind::TaskTerminalBatch, code,
                 first_task_id, last_task_id, 0u, 0u, -1, 0, 0, 0,
                 logical_count, first_ticket, last_ticket,
                 static_cast<std::uint64_t>(terminal_kind),
                 includes_root_submit ? 1u : 0u, 0u, first_task_id,
                 logical_tasks, canonical_hash, ReasonCode::Ok);
}

} // namespace rund::node
