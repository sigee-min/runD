#include "../../reactor/registry.hpp"
#include "../../state/model/task.hpp"
#include "../../state/model/timer.hpp"
#include "../../state/storage.hpp"
#include "pick.hpp"

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

ReadyPick
Scheduler::WaitUntilProgressReady(const std::uint64_t only_scope_id) noexcept {
  ReadyPick ready = ReadyPick::none();
  bool timer_slept = false;
  if (!state_->ready.timers.empty() &&
      ReactorRegistryEmpty(state_->reactor.reactor)) {
    std::this_thread::sleep_until(
        NextLogicalTimerWallDeadline(state_->ready.timers));
    ready = PopSubmittableReady(only_scope_id);
    timer_slept = true;
  }

  if (ready.disposition() == ReadyPickDisposition::Task ||
      ready.disposition() == ReadyPickDisposition::Activity) {
    return ready;
  }
  if (ReactorRegistryEmpty(state_->reactor.reactor)) {
    return ready.disposition() == ReadyPickDisposition::Blocked
               ? ready
               : (timer_slept ? ReadyPick::activity() : ReadyPick::none());
  }

  int timeout_ms = TimerBoundIoPollTimeoutMs();
  {
    std::lock_guard lock{state_->batches.direct_mutex};
    if (state_->batches.direct_jobs_in_flight != 0u) {
      timeout_ms = 0;
    }
  }
  const bool host_replay_failed_before = state_->identity.host_replay_failed;
  const bool reactor_activity = DrainReadyReactor(timeout_ms, true);
  if (!host_replay_failed_before && state_->identity.host_replay_failed) {
    return ready.disposition() == ReadyPickDisposition::Blocked
               ? ready
               : ReadyPick::activity();
  }
  ready = PopSubmittableReady(only_scope_id);
  if (ready.disposition() != ReadyPickDisposition::None) {
    return ready;
  }
  return timer_slept || reactor_activity ? ReadyPick::activity()
                                         : ReadyPick::none();
}

} // namespace rund::node
