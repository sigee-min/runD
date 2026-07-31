#include <rund/task/stats/slots.hpp>

#include "../../state/model/task.hpp"
#include "../../state/storage.hpp"

#include <rund/task/coroutine.hpp>

#include <utility>

namespace rund::node {

std::size_t Scheduler::ClaimSpawnTaskSlot(bool &reuse_record) noexcept {
  reuse_record = state_->ready.free_record_head != 0u;
  if (reuse_record) {
    const std::size_t record_index = state_->ready.free_record_head - 1u;
    TaskRecord &record = state_->ready.records[record_index];
    state_->ready.free_record_head =
        static_cast<std::uint32_t>(record.wake_ticket);
    record.wake_ticket = 0u;
    return record_index;
  }
  state_->ready.records.emplace_back();
  return state_->ready.records.size() - 1u;
}

void Scheduler::ReleasePreparedSpawnTaskSlot(TaskRecord &record,
                                             const std::size_t record_index,
                                             const bool reuse_record) noexcept {
  if (reuse_record) {
    record.recyclable = true;
    record.wake_ticket = state_->ready.free_record_head;
    state_->ready.free_record_head =
        static_cast<std::uint32_t>(record_index + 1u);
  } else {
    state_->ready.records.pop_back();
  }
}

bool Scheduler::MaterializePreparedSpawnTask(
    TaskRecord &record, ::rund::detail::task::Callable *const callable,
    const ::rund::detail::task::CoroutineStart coroutine,
    const CompletionLease completion, const std::uint64_t parent_task_id,
    const std::uint64_t scope_id, const bool root_spawn) noexcept {
  record = TaskRecord{};
  record.recyclable = false;
  record.id = IssueSpawnTaskId(parent_task_id, scope_id);
  if (record.id == 0u) {
    return false;
  }

  const bool consume_root_single_join_session_lane =
      !state_->lanes.lanes.empty() && root_spawn &&
      state_->batches.root_single_join_session_lane_active &&
      state_->batches.root_single_join_session_lane <
          state_->lanes.lanes.size() &&
      state_->ready.ready_depth == 0u;
  if (consume_root_single_join_session_lane) {
    record.home_lane = state_->batches.root_single_join_session_lane;
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::RootSingleJoinSessionLaneReuses);
    state_->batches.root_single_join_session_lane_active = false;
    state_->batches.root_single_join_session_lane = 0u;
  } else {
    if (!state_->lanes.lanes.empty() && root_spawn &&
        state_->batches.root_single_join_session_lane_active) {
      ++::rund::detail::task::Stat(
          state_->evidence.metrics,
          ::rund::detail::task::StatSlot::RootSingleJoinSessionLaneResets);
      state_->batches.root_single_join_session_lane_active = false;
      state_->batches.root_single_join_session_lane = 0u;
    }
    record.home_lane =
        state_->DefaultLaneIndexForTask(record.id, state_->lanes.lanes.size());
  }

  record.scope_id = scope_id;
  record.dynamic_scope_id = record.scope_id;
  ++state_->identity.next_spawn_index;
  record.coroutine_task = coroutine.frame != nullptr;
  if (record.coroutine_task) {
    record.coroutine_frame = coroutine.frame;
    record.coroutine_ops = coroutine.ops;
  }
  record.completion = CompletionPool::slot(completion);

  if (record.coroutine_task) {
    record.lane_segment_side_exit = true;
  }
  if (!record.coroutine_task && callable != nullptr &&
      callable->uses_inline_storage()) {
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::CallableInlineStores);
  }
  if (!record.coroutine_task && callable != nullptr) {
    record.callable =
        state_->resources.callable_pool.claim(std::move(*callable));
    if (record.callable == nullptr) {
      return false;
    }
  }
  return true;
}

} // namespace rund::node
