#include "../state/model/task.hpp"
#include "../state/storage.hpp"

#include <rund/task/stats/slots.hpp>

#include <algorithm>

namespace rund::node {

void Scheduler::FlushScopeBoundary() noexcept {
  FlushRootSingleJoinEpoch(ReasonCode::Ok);
  FlushTaskSpawnBatch(ReasonCode::Ok);
  FlushYieldBatch(ReasonCode::Ok);
  FlushPendingRootJoinRetireBatch();
}

bool Scheduler::InstallPlan(
    const ::rund::replay::detail::scope::Plan &plan) noexcept {
  if (!plan.valid() || state_->plan.installed || CurrentScopeId() != 1u) {
    return false;
  }
  FlushScopeBoundary();
  std::lock_guard lock{state_->evidence.mutex};
  state_->plan.value = plan;
  state_->plan.installed = true;
  state_->plan.bases = SchedulerPlanState::Bases{
      .task = state_->identity.next_task_id,
      .scope = state_->identity.next_scope_id,
      .wait = state_->identity.next_wait_id,
      .timer = state_->identity.next_timer_sequence,
      .observation = state_->identity.next_observation_sequence,
      .event = state_->identity.next_host_event_sequence,
      .channel = state_->identity.next_channel_id,
      .trace_epoch = state_->identity.next_trace_epoch_id,
      .ticket = state_->batches.next_commit_ticket_to_issue,
  };
  state_->plan.begin();
  if (state_->plan.bases.task == 0u || state_->plan.bases.scope == 0u ||
      state_->plan.bases.wait == 0u || state_->plan.bases.timer == 0u ||
      state_->plan.bases.observation == 0u || state_->plan.bases.event == 0u ||
      state_->plan.bases.channel == 0u ||
      state_->plan.bases.trace_epoch == 0u || state_->plan.bases.ticket == 0u) {
    state_->plan.fail();
  }
  ClearReplay();
  state_->identity.logical_time_ns.store(0, std::memory_order_release);
  return true;
}

void Scheduler::ClearPlan() noexcept {
  std::lock_guard lock{state_->evidence.mutex};
  state_->plan.value = {};
  state_->plan.installed = false;
  state_->plan.failure = ReasonCode::Ok;
  ClearReplay();
}

ScopeToken Scheduler::BeginScope() noexcept {
  if (CurrentTaskId() != 0u) {
    return ScopeToken{.code = RejectPrimitive()};
  }
  (void)TrapLaneOwnedSegmentPrimitive(
      ::rund::detail::task::OperationKind::ScopeEnter);
  EnsureCurrentCommit();
  const std::uint64_t previous = CurrentScopeId();
  constexpr std::uint64_t root_scope = 1u;
  if (previous == root_scope) {
    if (!state_->plan.installed) {
      FlushScopeBoundary();
    }
    std::lock_guard lock{state_->evidence.mutex};
    state_->evidence.metrics = ::rund::detail::task::StatStorage{};
    ::rund::detail::task::Stat(state_->evidence.metrics,
                               ::rund::detail::task::StatSlot::TaskWorkers) =
        std::max<std::uint32_t>(1u, state_->resources.limits.task_workers);
    ::rund::detail::task::Stat(state_->evidence.metrics,
                               ::rund::detail::task::StatSlot::TraceHash) =
        ::rund::detail::task::kTraceHashSeed;
    state_->resources.frame_arena.begin_epoch();
    RefreshResourceStats();
  }
  std::size_t observation_begin = 0u;
  std::size_t event_begin = 0u;
  {
    std::lock_guard lock{state_->evidence.mutex};
    if (state_->evidence.input_bytes == nullptr ||
        state_->evidence.input_bytes.use_count() != 1) {
      try {
        state_->evidence.input_bytes =
            std::make_shared<std::vector<std::byte>>(static_cast<std::size_t>(
                state_->resources.limits.host_payload_capacity_bytes));
      } catch (...) {
        return ScopeToken{.code = ReasonCode::TaskSchedulerAllocationFailed};
      }
    }
    state_->evidence.input_byte_size = 0u;
    if (previous == root_scope && !state_->plan.installed) {
      state_->evidence.input_consumed_bytes = 0u;
      state_->evidence.input_count = 0u;
    }
    observation_begin = state_->evidence.observations.size();
    event_begin = state_->evidence.host_events.size();
  }
  const std::uint64_t scope_id = state_->identity.next_scope_id++;
  SetCurrentScopeId(scope_id);
  Record(::rund::detail::task::OperationKind::ScopeEnter, ReasonCode::Ok,
         CurrentTaskId(), scope_id);
  return ScopeToken{
      .scope_id = scope_id,
      .previous_scope_id = previous,
      .observation_begin = observation_begin,
      .event_begin = event_begin,
      .code = ReasonCode::Ok,
  };
}

task::Status Scheduler::EndScope(const ScopeToken token) noexcept {
  if (token.code != ReasonCode::Ok) {
    return FailScope(token.code);
  }
  (void)TrapLaneOwnedSegmentPrimitive(
      ::rund::detail::task::OperationKind::ScopeWake);
  EnsureCurrentCommit();
  while (!state_->ScopeTerminal(token.scope_id)) {
    Record(::rund::detail::task::OperationKind::ScopePark, ReasonCode::Ok,
           CurrentTaskId(), token.scope_id);
    if (!Step(token.scope_id)) {
      if (state_->ScopeTerminal(token.scope_id)) {
        break;
      }
      if (WakeDeadlockedTasks(token.scope_id)) {
        continue;
      }
      SetCurrentScopeId(token.previous_scope_id);
      return FailScope(ReasonCode::TaskDeadlock);
    }
  }
  SetCurrentScopeId(token.previous_scope_id);
  TaskRecord *const active_record = state_->Find(CurrentTaskId());
  if (active_record != nullptr && active_record->state == TaskState::Running) {
    active_record->dynamic_scope_id = CurrentScopeId();
  }
  constexpr std::uint64_t root_scope = 1u;
  if (token.previous_scope_id == root_scope) {
    ValidateReplayDrain();
  }
  const ReasonCode code = token.previous_scope_id == root_scope
                              ? state_->FirstFailureCode()
                              : state_->ScopeFailureCode(token.scope_id);
  Record(::rund::detail::task::OperationKind::ScopeWake, code, CurrentTaskId(),
         token.scope_id);
  return code == ReasonCode::Ok ? task::Status::success()
                                : task::Status::fail(code);
}

} // namespace rund::node
