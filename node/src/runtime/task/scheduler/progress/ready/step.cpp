#include <rund/task/stats/slots.hpp>

#include "../../state/model/context.hpp"
#include "../../state/model/task.hpp"
#include "../../state/storage.hpp"
#include "pick.hpp"

#include <thread>

namespace rund::node {

bool Scheduler::DispatchReadyTask(const std::uint64_t id,
                                  const std::uint64_t only_scope_id) noexcept {
  bool dispatched = false;
  if (only_scope_id != 0u || active_scheduler_context != nullptr ||
      state_->lanes.lanes.empty()) {
    dispatched = RunOnLane(id);
  } else if (TaskRecord *const record = state_->Find(id);
             record != nullptr && record->lane_segment_side_exit) {
    dispatched = RunOnLane(id, 0u, true);
  } else {
    dispatched = RunMultiLaneReadyBatch(id);
  }
  if (dispatched) {
    return true;
  }

  // CanSubmitToLane() is a lock-separated observation. A worker can change
  // the lane between that probe and RunOnLane() taking the lane lock. Keep
  // the popped task authoritative until dispatch actually owns it; otherwise
  // the queue becomes empty while a live task remains Ready and the scheduler
  // reports a false deadlock.
  // A lane worker may be changing task storage while dispatch observes its
  // rejection. The one requeue owner tests the exact record under the
  // scheduler state lock before returning queue ownership.
  (void)RequeueReadyTask(id, only_scope_id);
  std::this_thread::yield();
  return true;
}

bool Scheduler::Step(const std::uint64_t only_scope_id) noexcept {
  if (only_scope_id == 0u && active_scheduler_context == nullptr) {
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::RootHotPathEntries);
  }

  const std::uint64_t failed_before = ::rund::detail::task::Stat(
      state_->evidence.metrics, ::rund::detail::task::StatSlot::Failed);
  ReadyPick ready = PopSubmittableReady(only_scope_id);
  if (ready.disposition() == ReadyPickDisposition::None) {
    if (state_->identity.host_replay_failed &&
        ::rund::detail::task::Stat(state_->evidence.metrics,
                                   ::rund::detail::task::StatSlot::Failed) !=
            failed_before) {
      return true;
    }
    ready = WaitUntilProgressReady(only_scope_id);
    if (state_->identity.host_replay_failed &&
        ::rund::detail::task::Stat(state_->evidence.metrics,
                                   ::rund::detail::task::StatSlot::Failed) !=
            failed_before) {
      return true;
    }
  }

  switch (ready.disposition()) {
  case ReadyPickDisposition::Task:
    return DispatchReadyTask(ready.task_id(), only_scope_id);
  case ReadyPickDisposition::Blocked:
    std::this_thread::yield();
    return true;
  case ReadyPickDisposition::Activity:
    return true;
  case ReadyPickDisposition::None:
    break;
  }

  if (WaitForDirectJobs()) {
    return true;
  }
  return false;
}

} // namespace rund::node
