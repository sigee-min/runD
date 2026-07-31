#include <rund/task/stats/slots.hpp>

#include "../../state/model/context.hpp"
#include "../../state/model/task.hpp"
#include "../../state/storage.hpp"
#include "../../reactor/registry.hpp"

namespace rund::node {

bool Scheduler::RunRootSingleJoinReadyTarget(const std::uint64_t id) noexcept {
  const auto reset_session = [this] {
    if (state_->batches.root_single_join_session_lane_active) {
      ++::rund::detail::task::Stat(
          state_->evidence.metrics,
          ::rund::detail::task::StatSlot::RootSingleJoinSessionLaneResets);
    }
    state_->batches.root_single_join_session_lane_active = false;
    state_->batches.root_single_join_session_lane = 0u;
  };
  if (id == 0u || CurrentTaskId() != 0u ||
      active_scheduler_context != nullptr) {
    return false;
  }
  if (!state_->ready.timers.empty() ||
      !ReactorRegistryEmpty(state_->reactor.reactor) ||
      state_->batches.direct_jobs_in_flight != 0u) {
    state_->batches.root_single_join_streak = 0u;
    reset_session();
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::RootJoinReadyFastPathMisses);
    return false;
  }
  if (ReadyDepth() != 1u) {
    state_->batches.root_single_join_streak = 0u;
    reset_session();
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::RootJoinReadyFastPathMisses);
    return false;
  }
  TaskRecord *const record = state_->Find(id);
  if (record == nullptr || record->state != TaskState::Ready) {
    state_->batches.root_single_join_streak = 0u;
    reset_session();
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::RootJoinReadyFastPathMisses);
    return false;
  }
  const std::uint64_t ready_id = PopReady(0u);
  if (ready_id != id) {
    if (ready_id != 0u) {
      RestoreReadyFront(ready_id, 0u);
    }
    state_->batches.root_single_join_streak = 0u;
    reset_session();
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::RootJoinReadyFastPathMisses);
    return false;
  }
  const bool hot_standby = state_->batches.root_single_join_streak != 0u;
  if (RunOnLane(id, 0u, false, true, hot_standby)) {
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::RootJoinReadyFastPaths);
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::RootJoinCommitTicketsAvoided);
    ++state_->batches.root_single_join_streak;
    state_->batches.root_single_join_session_lane =
        state_->LaneIndexForTask(record, state_->lanes.lanes.size());
    state_->batches.root_single_join_session_lane_active = true;
    return true;
  }
  RestoreReadyFront(id, 0u);
  state_->batches.root_single_join_streak = 0u;
  reset_session();
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::RootJoinReadyFastPathMisses);
  return false;
}

} // namespace rund::node
