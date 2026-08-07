#include <rund/counter.hpp>
#include <rund/task/stats/slots.hpp>

#include "../../../state/storage/check.hpp"
#include "local.hpp"

namespace rund::node {

void Scheduler::RecordLaneLocalYieldBatch(
    const std::uint64_t task_id, const std::uint64_t logical_yields,
    const std::uint64_t logical_ticket) noexcept {
  state_->RequireSequencer();
  if (task_id == 0u || logical_yields == 0u) {
    return;
  }
  FlushRootSingleJoinEpoch(ReasonCode::Ok);
  FlushTaskSpawnBatch(ReasonCode::Ok);
  FlushYieldBatch(ReasonCode::Ok);
  const std::uint64_t order_hash = YieldBatchOrderHash(
      false, state_->plan.task(task_id), logical_yields, logical_yields);
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::YieldBatchPackets);
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::YieldBatchLogicalEvents) +=
      logical_yields;
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::YieldBatchOrderHashFastPaths);
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::YieldLaneLocalReadyRingBatches);
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::YieldLaneLocalReadyRingLogicalYields) +=
      logical_yields;
  ::rund::detail::counter::Accumulate(
      ::rund::detail::task::Stat(state_->evidence.metrics,
                                 ::rund::detail::task::StatSlot::Yields),
      logical_yields);
  ::rund::detail::task::Stat(state_->evidence.metrics,
                             ::rund::detail::task::StatSlot::Parked) +=
      logical_yields;
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::YieldLaneLocalReadyRingPushes) +=
      logical_yields;
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::YieldLaneLocalReadyRingPops) +=
      logical_yields;
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::YieldLaneLocalReadyRingSwitches) +=
      logical_yields;
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::TerminalYieldRejections) +=
      logical_yields;
  RecordPhysical(::rund::detail::task::OperationKind::YieldBatch,
                 ReasonCode::Ok, task_id, task_id, 0u, 0u, -1, 0, 0, 0,
                 logical_yields, logical_ticket, logical_ticket, logical_yields,
                 0u, 0u, 0u, logical_yields, order_hash, ReasonCode::Ok);
}

void Scheduler::RecordLaneLocalYieldEpochBatch(
    const std::uint64_t first_task_id, const std::uint64_t last_task_id,
    const std::uint64_t first_ticket, const std::uint64_t last_ticket,
    const std::uint64_t logical_tasks, const std::uint64_t logical_yields,
    const std::uint64_t participant_hash) noexcept {
  state_->RequireSequencer();
  if (first_task_id == 0u || last_task_id < first_task_id ||
      logical_tasks == 0u || logical_yields == 0u) {
    return;
  }
  FlushRootSingleJoinEpoch(ReasonCode::Ok);
  FlushTaskSpawnBatch(ReasonCode::Ok);
  FlushYieldBatch(ReasonCode::Ok);
  const std::uint64_t order_hash = LaneLocalYieldEpochOrderHash(
      state_->plan.task(first_task_id), state_->plan.task(last_task_id),
      state_->plan.ticket(first_ticket), state_->plan.ticket(last_ticket),
      logical_tasks, logical_yields, participant_hash);
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::YieldBatchPackets);
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::YieldBatchLogicalEvents) +=
      logical_yields;
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::YieldBatchOrderHashFastPaths);
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::YieldLaneLocalReadyRingBatches);
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::YieldLaneLocalReadyRingLogicalYields) +=
      logical_yields;
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::YieldLaneLocalReadyRingPushes) +=
      logical_yields;
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::YieldLaneLocalReadyRingPops) +=
      logical_yields;
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::YieldLaneLocalReadyRingSwitches) +=
      logical_yields;
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::YieldLaneLocalEpochPackets);
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::YieldLaneLocalEpochLogicalTasks) +=
      logical_tasks;
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::YieldLaneLocalEpochLogicalYields) +=
      logical_yields;
  ::rund::detail::counter::Accumulate(
      ::rund::detail::task::Stat(state_->evidence.metrics,
                                 ::rund::detail::task::StatSlot::Yields),
      logical_yields);
  ::rund::detail::task::Stat(state_->evidence.metrics,
                             ::rund::detail::task::StatSlot::Parked) +=
      logical_yields;
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::TerminalYieldRejections) +=
      logical_yields;
  RecordPhysical(::rund::detail::task::OperationKind::YieldBatch,
                 ReasonCode::Ok, first_task_id, last_task_id, 0u, 0u, -1, 0, 0,
                 0, logical_yields, first_ticket, last_ticket, logical_yields,
                 0u, 0u, 0u, logical_yields, order_hash, ReasonCode::Ok);
}

} // namespace rund::node
