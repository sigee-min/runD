#include <rund/task/stats/slots.hpp>

#include "../../state/model/context.hpp"
#include "../../state/storage.hpp"

#include <algorithm>

namespace rund::node {

std::uint64_t
Scheduler::IssueSpawnTaskId(const std::uint64_t parent_task_id,
                            const std::uint64_t scope_id) noexcept {
  const std::uint64_t range_size = std::max<std::uint64_t>(
      1u,
      std::min<std::uint64_t>(state_->resources.limits.task_capacity,
                              state_->resources.limits.ready_queue_capacity));
  const bool root_range_candidate =
      parent_task_id == 0u && (active_scheduler_context == nullptr ||
                               active_scheduler_context->task_id == 0u);
  auto &reservation = state_->batches.spawn_id_reservation;
  if (!root_range_candidate) {
    reservation = SchedulerBatchState::SpawnIdReservation{};
    return state_->identity.next_task_id++;
  }

  const bool reservation_matches =
      reservation.active && reservation.remaining != 0u &&
      reservation.parent_task_id == parent_task_id &&
      reservation.scope_id == scope_id;
  if (!reservation_matches) {
    const std::uint64_t range_first = state_->identity.next_task_id;
    const std::uint64_t range_end = range_first + range_size;
    reservation.active = true;
    reservation.parent_task_id = parent_task_id;
    reservation.scope_id = scope_id;
    reservation.next_task_id = range_first;
    reservation.remaining = range_size;
    reservation.index_materialized = false;
    state_->identity.next_task_id = range_end;
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::SpawnTaskIdRangeReservations);
    ::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::SpawnTaskIdRangeReservedSlots) +=
        range_size;
  } else {
    if (!reservation.index_materialized && state_->ready.ready_depth != 0u) {
      reservation.index_materialized = true;
      ++::rund::detail::task::Stat(
          state_->evidence.metrics,
          ::rund::detail::task::StatSlot::SpawnEpochMaterializations);
      ::rund::detail::task::Stat(
          state_->evidence.metrics,
          ::rund::detail::task::StatSlot::SpawnEpochMaterializedTasks) +=
          reservation.remaining;
    }
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::SpawnPerTaskIdAllocationsAvoided);
  }

  const std::uint64_t id = reservation.next_task_id++;
  --reservation.remaining;
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::SpawnTaskIdRangeUsedSlots);
  if (reservation.remaining == 0u) {
    reservation = SchedulerBatchState::SpawnIdReservation{};
  }
  return id;
}

} // namespace rund::node
