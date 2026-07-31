#include <rund/task/stats/slots.hpp>

#include "../../state/model/task.hpp"
#include "../../state/storage.hpp"

#include <algorithm>
#include <cstdlib>

namespace rund::node {

std::size_t Scheduler::ReadyDepth() const noexcept {
  std::lock_guard lock{state_->evidence.mutex};
  return state_->ready.ready_depth;
}

bool Scheduler::ReadyQueuesEmpty() const noexcept { return ReadyDepth() == 0u; }

std::uint64_t Scheduler::PopReady(const std::uint64_t only_scope_id) noexcept {
  WakeDueTimers();
  WakeReadyReactor();
  std::lock_guard lock{state_->evidence.mutex};
  if (only_scope_id == 0u) {
    while (!state_->ready.ready.empty()) {
      std::uint64_t id = 0u;
      (void)state_->ready.ready.pop_front(id);
      if (state_->ready.ready_depth != 0u) {
        --state_->ready.ready_depth;
      }
      const TaskRecord *const record = state_->Find(id);
      if (record != nullptr && record->state == TaskState::Ready) {
        ++::rund::detail::task::Stat(
            state_->evidence.metrics,
            ::rund::detail::task::StatSlot::GlobalReadyQueuePops);
        return id;
      }
    }
    return 0u;
  }

  const auto take_scope = [this, only_scope_id](auto &queue) {
    std::uint64_t id = 0u;
    const bool found = queue.take_first(
        [&](const auto candidate) {
          const auto id = candidate;
          const TaskRecord *const record = state_->Find(id);
          return record != nullptr && record->state == TaskState::Ready &&
                 record->scope_id == only_scope_id;
        },
        id);
    if (!found)
      return std::uint64_t{0u};
    if (state_->ready.ready_depth != 0u) {
      --state_->ready.ready_depth;
    }
    return id;
  };
  if (const std::uint64_t id = take_scope(state_->ready.ready); id != 0u) {
    return id;
  }
  return 0u;
}

void Scheduler::RestoreReadyFront(const std::uint64_t id,
                                  const std::uint64_t) noexcept {
  std::lock_guard lock{state_->evidence.mutex};
  const bool restored = state_->ready.ready.push_front(id);
  // Restore follows a successful pop, so the same bounded queue must have one
  // free slot. Continuing would corrupt ready_depth and lose runnable work.
  if (!restored) {
    std::abort();
  }
  ++state_->ready.ready_depth;
  ::rund::detail::task::Stat(state_->evidence.metrics,
                             ::rund::detail::task::StatSlot::MaxReadyDepth) =
      std::max<std::uint64_t>(
      ::rund::detail::task::Stat(
              state_->evidence.metrics,
              ::rund::detail::task::StatSlot::MaxReadyDepth),
          state_->ready.ready_depth);
}

bool Scheduler::RequeueReadyTask(const std::uint64_t id,
                                 const std::uint64_t only_scope_id) noexcept {
  std::lock_guard lock{state_->evidence.mutex};
  TaskRecord *const record = state_->Find(id);
  if (record == nullptr || record->state != TaskState::Ready) {
    return false;
  }
  RestoreReadyFront(id, only_scope_id);
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::LaneDispatchRequeuedTasks);
  return true;
}

Scheduler::ReadyPick
Scheduler::PopSubmittableReady(const std::uint64_t only_scope_id) noexcept {
  if (only_scope_id == 0u) {
    std::uint64_t head = 0u;
    bool semantic_head = false;
    {
      std::lock_guard lock{state_->evidence.mutex};
      if (state_->ready.ready.front(head)) {
        const TaskRecord *const record = state_->Find(head);
        semantic_head = record != nullptr &&
                        record->state == TaskState::Ready &&
                        record->lane_segment_side_exit;
      }
    }
    if (semantic_head && !CanSubmitToLane(head)) {
      return ReadyPick{.id = 0u, .blocked = true};
    }
  }

  const std::uint64_t id = PopReady(only_scope_id);
  if (id == 0u || CanSubmitToLane(id)) {
    return ReadyPick{.id = id};
  }
  const TaskRecord *const record = state_->Find(id);
  if (record != nullptr && !record->lane_segment_side_exit &&
      !record->coroutine_task && DispatchQueuedReady(id)) {
    return ReadyPick{.activity = true};
  }
  RestoreReadyFront(id, only_scope_id);
  return ReadyPick{.id = 0u, .blocked = true};
}

} // namespace rund::node
