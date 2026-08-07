#include <rund/counter.hpp>
#include <rund/task/stats/slots.hpp>

#include "../../state/segment.hpp"
#include "../../state/model/task.hpp"
#include "../../state/storage.hpp"

namespace rund::node {

namespace {

[[nodiscard]] bool
LaneOwnedSegmentTaskIdsContiguous(const std::vector<std::uint64_t> &task_ids,
                                  const std::size_t logical_tasks) noexcept {
  if (logical_tasks == 0u || logical_tasks != task_ids.size()) {
    return false;
  }
  for (std::size_t index = 1u; index < task_ids.size(); ++index) {
    if (task_ids[index] != task_ids[index - 1u] + 1u) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool
LaneOwnedSegmentTicketsContiguous(const std::uint64_t first_ticket,
                                  const std::uint64_t last_ticket,
                                  const std::size_t logical_tasks) noexcept {
  if (logical_tasks == 0u || first_ticket == 0u || last_ticket < first_ticket) {
    return false;
  }
  return last_ticket - first_ticket + 1u == logical_tasks;
}

} // namespace

bool Scheduler::CommitSuccessfulLaneOwnedSegments(
    const std::vector<std::uint64_t> &task_ids, const std::size_t logical_tasks,
    const LaneOwnedSegmentSummary &summary) noexcept {
  if (!summary.success || logical_tasks == 0u || task_ids.empty()) {
    return false;
  }
  if (!LaneOwnedSegmentTaskIdsContiguous(task_ids, logical_tasks) ||
      !LaneOwnedSegmentTicketsContiguous(summary.first_ticket,
                                         summary.last_ticket, logical_tasks)) {
    ::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::DeterministicCommitBatches) +=
        summary.submitted_lanes;
    ::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::DeterministicCommitLogicalEvents) +=
        summary.commit_logical_events;
    return false;
  }
  std::uint64_t order_hash = kFnvOffset;
  for (std::size_t index = 0u; index < task_ids.size(); ++index) {
    const std::uint64_t task_id = task_ids[index];
    TaskRecord *const record = state_->Find(task_id);
    if (record == nullptr || record->id != task_id ||
        record->state != TaskState::Running) {
      std::abort();
    }
    record->state = TaskState::Completed;
    record->failure_code = ReasonCode::Ok;
    record->quantum_active = false;
    DestroyLaneCallable(*record);
    MixHash(order_hash, state_->plan.task(task_id));
    MixHash(order_hash, state_->plan.ticket(summary.first_ticket + index));
    MixHash(order_hash, static_cast<std::uint64_t>(
                            ::rund::detail::task::OperationKind::Complete));
    MixHash(order_hash, static_cast<std::uint64_t>(ReasonCode::Ok));
    MixHash(order_hash, index + 1u);
  }
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::DeterministicCommitBatches);
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::DeterministicCommitLogicalEvents) +=
      summary.commit_logical_events;
  ::rund::detail::counter::Accumulate(
      ::rund::detail::task::Stat(state_->evidence.metrics,
                                 ::rund::detail::task::StatSlot::Completed),
      logical_tasks);
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::LaneLocalTasksCompleted) += logical_tasks;
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::LaneLocalRetireCandidates) +=
      logical_tasks;
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::DeterministicEffectLogMergeBatches);
  ::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::EffectEventsMerged) += logical_tasks;
  RecordTerminalRangeBatch(::rund::detail::task::OperationKind::Complete,
                           ReasonCode::Ok, task_ids.front(), task_ids.back(),
                           summary.first_ticket, summary.last_ticket,
                           logical_tasks, order_hash, true);
  if (!state_->ready.join_waits.empty()) {
    for (const std::uint64_t task_id : task_ids) {
      WakeJoinWaiters(task_id, ReasonCode::Ok);
    }
  }
  return true;
}

} // namespace rund::node
