#include "../../state/model/task.hpp"
#include "../../state/model/timer.hpp"
#include "../../state/storage.hpp"
#include "../../reactor/registry.hpp"

#include <algorithm>
#include <thread>

namespace rund::node {
namespace {

[[nodiscard]] Clock::time_point
NextLogicalTimerWallDeadline(const std::vector<TimerWait> &timers) noexcept {
  return timers.front().deadline;
}

} // namespace

bool Scheduler::WakeDeadlockedTasks(
    const std::uint64_t only_scope_id) noexcept {
  std::vector<TaskRecord *> victims{};
  for (TaskRecord &record : state_->ready.records) {
    if (record.state != TaskState::ChannelBlocked) {
      continue;
    }
    if (only_scope_id != 0u && record.scope_id != only_scope_id) {
      continue;
    }
    victims.push_back(&record);
  }
  if (victims.empty()) {
    return false;
  }
  std::sort(victims.begin(), victims.end(),
            [](const TaskRecord *const lhs, const TaskRecord *const rhs) {
              if (lhs->wait_id != rhs->wait_id) {
                return lhs->wait_id < rhs->wait_id;
              }
              return lhs->id < rhs->id;
            });
  for (TaskRecord *const record : victims) {
    if (record == nullptr || record->state != TaskState::ChannelBlocked) {
      continue;
    }
    const std::uint64_t wait_id = record->wait_id;
    const std::uint64_t channel_id = record->wait_source_id;
    record->wait_result = ReasonCode::TaskDeadlock;
    record->state = TaskState::Ready;
    record->lane_segment_side_exit = true;
    if (state_->resources.live_channel_waits > 0u) {
      --state_->resources.live_channel_waits;
    }
    state_->EnqueueProgress(*record);
    Record(::rund::detail::task::OperationKind::DeadlockWake,
           ReasonCode::TaskDeadlock, record->id, 0u, wait_id, channel_id);
  }
  return true;
}

bool Scheduler::WaitForDirectJobs() noexcept {
  std::unique_lock<std::mutex> lock(state_->batches.direct_mutex);
  if (state_->batches.ready_epoch != state_->batches.consumed_ready_epoch) {
    state_->batches.consumed_ready_epoch = state_->batches.ready_epoch;
    return true;
  }
  if (state_->batches.direct_jobs_in_flight == 0u) {
    return false;
  }
  state_->batches.direct_cv.wait(lock, [this] {
    return state_->batches.direct_jobs_in_flight == 0u ||
           state_->batches.ready_epoch != state_->batches.consumed_ready_epoch;
  });
  state_->batches.consumed_ready_epoch = state_->batches.ready_epoch;
  return true;
}

bool Scheduler::WaitUntilTimerReady(const std::uint64_t only_scope_id,
                                    ReadyPick *const ready) noexcept {
  if (ready->id != 0u || ready->activity || state_->ready.timers.empty() ||
      !ReactorRegistryEmpty(state_->reactor.reactor)) {
    return false;
  }
  std::this_thread::sleep_until(
      NextLogicalTimerWallDeadline(state_->ready.timers));
  *ready = PopSubmittableReady(only_scope_id);
  return true;
}

} // namespace rund::node
