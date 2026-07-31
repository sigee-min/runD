#include "../../state/model/lane.hpp"
#include "../../state/model/task.hpp"
#include "../../state/storage.hpp"
#include "../../lane/segment/transaction.hpp"

#include <algorithm>
#include <cstdlib>

namespace rund::node {

task::Handle Scheduler::CurrentHandle() noexcept {
  EnsureCurrentCommit();
  TaskRecord *const record = state_->Find(CurrentTaskId());
  if (record == nullptr) {
    return InvalidHandle(ReasonCode::TaskContextMissing);
  }
  return ::rund::detail::task::HandleAccess::Make(
      record->id, state_->identity.scheduler_id, record->scope_id,
      ReasonCode::Ok);
}

bool Scheduler::DispatchExternal(const task::Handle &handle) noexcept {
  TaskRecord *const record = state_->Find(handle.id());
  if (record == nullptr || !Matches(handle, record) ||
      record->state != TaskState::Ready || state_->lanes.lanes.empty()) {
    return false;
  }

  std::vector<std::uint64_t> &admitted =
      state_->lanes.ready_deferred;
  if (admitted.capacity() < state_->resources.limits.task_capacity) {
    std::abort();
  }
  admitted.clear();
  for (;;) {
    const std::uint64_t id = PopReady(0u);
    if (id == 0u) {
      break;
    }
    if (admitted.size() == admitted.capacity()) {
      std::abort();
    }
    admitted.push_back(id);
    if (id == handle.id()) {
      break;
    }
  }
  if (admitted.empty() || admitted.back() != handle.id()) {
    for (std::size_t reverse = admitted.size(); reverse > 0u; --reverse) {
      RestoreReadyFront(admitted[reverse - 1u], 0u);
    }
    return false;
  }
  std::vector<std::uint8_t> &notify =
      state_->lanes.lane_batch_used_scratch;
  if (notify.size() != state_->lanes.lanes.size()) {
    std::abort();
  }
  std::fill(notify.begin(), notify.end(), std::uint8_t{0u});

  bool published = true;
  {
    segment::Locks locks{state_->lanes.lanes};
    for (const std::uint64_t id : admitted) {
      TaskRecord *const candidate = state_->Find(id);
      const std::size_t lane_index =
          state_->LaneIndexForTask(candidate, state_->lanes.lanes.size());
      if (candidate == nullptr || candidate->state != TaskState::Ready ||
          lane_index >= state_->lanes.lanes.size() ||
          state_->lanes.lanes[lane_index]->stop) {
        published = false;
        break;
      }
    }
    if (published) {
      for (const std::uint64_t id : admitted) {
        if (PrepareLaneQuantum(id) == nullptr) {
          std::abort();
        }
      }
      const std::uint64_t first_ticket = IssueCommitTickets(admitted.size());
      if (first_ticket == 0u) {
        std::abort();
      }
      for (std::size_t index = 0u; index < admitted.size(); ++index) {
        TaskRecord *const candidate = state_->Find(admitted[index]);
        if (candidate == nullptr) {
          std::abort();
        }
        const std::size_t lane_index =
            state_->LaneIndexForTask(candidate, state_->lanes.lanes.size());
        TaskLane &lane = *state_->lanes.lanes[lane_index];
        const std::uint64_t sequence = lane.next_job_sequence++;
        QueueDirect(lane, *candidate, first_ticket + index, sequence);
        lane.ready_signal.store(sequence, std::memory_order_release);
        notify[lane_index] = lane.ready_wait_requested ? 1u : 0u;
      }
    }
  }
  if (!published) {
    for (std::size_t reverse = admitted.size(); reverse > 0u; --reverse) {
      RestoreReadyFront(admitted[reverse - 1u], 0u);
    }
    return false;
  }
  for (std::size_t index = 0u; index < notify.size(); ++index) {
    if (notify[index] != 0u) {
      state_->lanes.lanes[index]->ready.notify_one();
    }
  }
  return true;
}

} // namespace rund::node
