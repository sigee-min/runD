#include <rund/task/stats/slots.hpp>

#include "../reactor/backend.hpp"
#include "../state/model/join.hpp"
#include "../state/model/stop.hpp"
#include "../state/model/task.hpp"
#include "../state/model/timer.hpp"
#include "../state/storage.hpp"

#include <algorithm>

namespace rund::node {

::rund::host::random::RunSeed Scheduler::RandomSeed() const noexcept {
  return state_->identity.random_seed;
}

void Scheduler::Reset() noexcept {
  StopHostIo();
  StopLanes();
  for (TaskRecord &record : state_->ready.records) {
    DestroyTask(record);
  }
  state_->ready.ready.clear();
  state_->ready.ready_depth = 0u;
  state_->ready.records.clear();
  state_->ready.record_index_slots.clear();
  state_->ready.record_index_size = 0u;
  state_->ready.record_index_deleted = 0u;
  state_->ready.free_record_head = 0u;
  state_->batches.pending_root_join_range =
      SchedulerBatchState::PendingRootJoinRange{};
  state_->ready.timers.clear();
  state_->ready.timer_wait_id_index.clear();
  state_->ready.join_waits.clear();
  state_->reactor.reactor_many_groups.clear();
  state_->reactor.reactor_many_requests.clear();
  state_->reactor.reactor_many_request_scratch.clear();
  state_->reactor.reactor_many_index_scratch.clear();
  state_->reactor.reactor_many_group_id_scratch.clear();
  state_->reactor.reactor_many_event_slots.clear();
  state_->reactor.reactor_many_event_slots_scratch.clear();
  state_->reactor.reactor_many_poll_request_scratch.clear();
  state_->reactor.reactor_many_ready_result_scratch.clear();
  state_->reactor.reactor_ready_code_scratch.clear();
  state_->reactor.reactor_socket_lease_scratch.clear();
  state_->reactor.reactor_ready_sets.clear();
  state_->reactor.reactor_many_validation_comparisons = 0u;
  state_->reactor.reactor_many_request_copies = 0u;
  state_->reactor.reactor_many_storage_growths = 0u;
  state_->reactor.reactor_ready_set_storage_growths = 0u;
  state_->reactor.stop_sources.clear();
  state_->reactor.canceled_wait_scratch.clear();
  ReactorCloseRuntime(state_->reactor.reactor);
  ClearReplay();
  state_->plan = {};
  state_->evidence.input_bytes.reset();
  state_->evidence.next_input_capture_token = 1u;
  state_->reactor.reactor_host_event_scratch.clear();
  state_->resources.callable_pool.reset();
  state_->resources.frame_arena.reset();
  state_->resources.completion_pool.reset();
  state_->resources.live_tasks.store(0u, std::memory_order_relaxed);
  state_->identity.next_task_id = 1u;
  state_->identity.next_scope_id = 2u;
  state_->identity.next_wait_id = 1u;
  state_->identity.next_reactor_many_group_id = 1u;
  state_->identity.next_reactor_ready_set_id = 1u;
  state_->identity.next_stop_source_id = 1u;
  ++state_->identity.stop_source_epoch;
  if (state_->identity.stop_source_epoch == 0u) {
    state_->identity.stop_source_epoch = 1u;
  }
  state_->identity.next_timer_sequence = 1u;
  state_->identity.next_observation_sequence = 1u;
  state_->identity.next_host_event_sequence = 1u;
  state_->identity.logical_time_ns.store(0, std::memory_order_release);
  state_->identity.next_channel_id = 1u;
  state_->identity.next_trace_epoch_id = 1u;
  state_->batches.root_single_join_streak = 0u;
  state_->resources.live_channels = 0u;
  state_->resources.live_channel_buffer_slots = 0u;
  state_->resources.live_channel_waits = 0u;
  state_->identity.active_scope_id = 1u;
  state_->identity.active_task_id = 0u;
  state_->identity.next_spawn_index = 1u;
  state_->batches.next_commit_ticket = 1u;
  state_->batches.next_commit_ticket_to_issue = 1u;
  state_->batches.lane_commit_first.store(0u, std::memory_order_relaxed);
  state_->batches.lane_commit_frontier.store(0u, std::memory_order_relaxed);
  state_->batches.lane_commit_end.store(0u, std::memory_order_relaxed);
  state_->batches.lane_commit_waiters.store(0u, std::memory_order_relaxed);
  state_->batches.lane_commit_active.store(false, std::memory_order_relaxed);
  state_->batches.direct_jobs_in_flight = 0u;
  state_->batches.task_direct_jobs_in_flight = 0u;
  state_->batches.ready_epoch = 0u;
  state_->batches.consumed_ready_epoch = 0u;
  state_->evidence.metrics = ::rund::detail::task::StatStorage{};
  state_->batches.task_spawn_batch = SchedulerBatchState::TaskSpawnBatch{};
  state_->batches.root_single_join_epoch =
      SchedulerBatchState::RootSingleJoinEpoch{};
  state_->batches.yield_resume_batch = SchedulerBatchState::YieldResumeBatch{};
  state_->batches.spawn_id_reservation =
      SchedulerBatchState::SpawnIdReservation{};
  state_->batches.lane_residual_join_owner_policy =
      SchedulerBatchState::LaneResidualJoinOwnerPolicy{};
  ::rund::detail::task::Stat(state_->evidence.metrics,
                             ::rund::detail::task::StatSlot::TaskWorkers) =
      std::max<std::uint32_t>(1u, state_->resources.limits.task_workers);
  ::rund::detail::task::Stat(state_->evidence.metrics,
                             ::rund::detail::task::StatSlot::TraceHash) =
      ::rund::detail::task::kTraceHashSeed;
}

} // namespace rund::node
