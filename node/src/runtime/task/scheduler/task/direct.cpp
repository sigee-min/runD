#include <rund/task/stats/slots.hpp>

#include "../state/model/context.hpp"
#include "../state/model/lane.hpp"
#include "../state/model/task.hpp"
#include "../state/storage.hpp"

#include <algorithm>
#include <mutex>

namespace rund::node {

bool Scheduler::TrapLaneOwnedSegmentPrimitive(
    const ::rund::detail::task::OperationKind kind,
    const ReasonCode code) noexcept {
  SchedulerThreadContext *const context = active_scheduler_context;
  if (context == nullptr || context->scheduler != this ||
      !context->lane_owned_segment_active ||
      context->lane_owned_segment_trapped) {
    return false;
  }
  TaskRecord *const record = static_cast<TaskRecord *>(context->record);
  if (record == nullptr || record->state != TaskState::Running) {
    return false;
  }
  context->lane_owned_segment_active = false;
  context->lane_owned_segment_trapped = true;
  if (context->lane_effect != nullptr) {
    context->lane_effect->trapped = true;
    context->lane_effect->trap_kind = kind;
    context->lane_effect->trap_code = code;
  }
  return true;
}

bool Scheduler::TryResumeSameLane(TaskRecord &record,
                                  SchedulerThreadContext &context) noexcept {
  if (active_task_lane == nullptr || context.scheduler != this ||
      context.lane_owned_segment_active || context.lane_owned_segment_trapped ||
      !context.commit_acquired || context.commit_ticket == 0u ||
      state_->lanes.lanes.empty() || record.state != TaskState::Ready ||
      ReadyQueuesEmpty()) {
    return false;
  }
  const std::uint64_t ready_id = PopReady(0u);
  if (ready_id != record.id) {
    if (ready_id != 0u) {
      RestoreReadyFront(ready_id, 0u);
    }
    return false;
  }
  const std::size_t lane_index =
      state_->LaneIndexForTask(&record, state_->lanes.lanes.size());
  if (lane_index >= state_->lanes.lanes.size() ||
      active_task_lane != state_->lanes.lanes[lane_index].get()) {
    return false;
  }
  record.state = TaskState::Running;
  {
    std::lock_guard<std::mutex> lock(state_->batches.commit_mutex);
    if (state_->batches.next_commit_ticket == context.commit_ticket) {
      ++state_->batches.next_commit_ticket;
    }
    context.commit_ticket = state_->batches.next_commit_ticket_to_issue++;
  }
  context.commit_acquired = false;
  context.root_submit_recorded = false;
  context.pending_root_submit = false;
  ++::rund::detail::task::Stat(state_->evidence.metrics,
                               ::rund::detail::task::StatSlot::SameLaneResumes);
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::SchedulerContextReuseHits);
  state_->batches.commit_cv.notify_all();
  EnsureCurrentCommit();
  return true;
}

} // namespace rund::node
