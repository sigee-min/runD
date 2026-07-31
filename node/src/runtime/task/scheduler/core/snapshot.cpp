#include "../../stats/access.hpp"
#include "../access.hpp"
#include "../reactor/ready/set/store.hpp"
#include "../state/storage.hpp"
#include "../state/storage/check.hpp"
#include "../state/task/commit.hpp"

#include "../../scope/evidence.hpp"
#include <rund/net/limits.hpp>
#include <rund/task/stats.hpp>

#include <algorithm>
#include <vector>

namespace rund::node {

::rund::SchedulerConfig scheduler_access::ActiveLimits() noexcept {
  const Scheduler *const scheduler = Scheduler::Active();
  if (scheduler == nullptr) {
    return ::rund::SchedulerConfig{};
  }
  return scheduler->state_->resources.limits;
}

ScopeEvidence Scheduler::CaptureScope(const std::size_t observation_begin,
                                      const std::size_t event_begin) {
  ControlCommitScope commit{*this};
  std::lock_guard lock{state_->evidence.mutex};
  state_->RequireSequencer();
  FlushRootSingleJoinEpoch(ReasonCode::Ok);
  FlushTaskSpawnBatch(ReasonCode::Ok);
  FlushYieldBatch(ReasonCode::Ok);
  FlushPendingRootJoinRetireBatch();
  RefreshResourceStats();

  const auto observation_first =
      state_->evidence.observations.begin() +
      std::min(observation_begin, state_->evidence.observations.size());
  const auto event_first =
      state_->evidence.host_events.begin() +
      std::min(event_begin, state_->evidence.host_events.size());
  return ScopeEvidence{
      .tasks =
          ::rund::detail::task::StatsAccess::Snapshot(state_->evidence.metrics),
      .memory = state_->resources.prepared_memory,
      .observations = {observation_first, state_->evidence.observations.end()},
      .events = {event_first, state_->evidence.host_events.end()},
      .payloads = CapturePayloads(),
      .input_rows = state_->evidence.input_count,
      .input_bytes = state_->evidence.input_consumed_bytes,
      .ready_capacity = state_->resources.limits.ready_queue_capacity,
  };
}

::rund::net::Limits Scheduler::ReadLimits() noexcept {
  std::lock_guard lock{state_->evidence.mutex};
  (void)TrapLaneOwnedSegmentPrimitive();
  EnsureCurrentCommit();
  const std::uint32_t ready_sets = static_cast<std::uint32_t>(
      std::count_if(state_->reactor.reactor_ready_sets.begin(),
                    state_->reactor.reactor_ready_sets.end(),
                    [](const ReactorReadySet &set) { return set.live; }));
  const std::uint32_t ready_set_members =
      ReactorReadySetMemberCount(state_->reactor.reactor_ready_sets);
  ::rund::net::Limits limits{ReasonCode::Ok};
  limits.ready_sets = ready_sets;
  limits.ready_set_members = ready_set_members;
  limits.max_ready_sets = state_->resources.limits.net_ready_set_capacity;
  limits.max_ready_set_members =
      state_->resources.limits.net_ready_set_member_capacity;
  limits.max_iov = state_->resources.limits.net_iov_capacity;
  limits.max_datagram_bytes =
      state_->resources.limits.net_datagram_capacity_bytes;
  limits.max_socket_registry_entries =
      state_->resources.limits.net_socket_registry_capacity;
  CompletePrimitiveCommit();
  return limits;
}

::rund::node::replay_detail::payload::Archive
Scheduler::CapturePayloads() const {
  if (state_->plan.mode() == ::rund::replay::detail::scope::Mode::Replay &&
      !state_->identity.host_replay_failed &&
      !state_->identity.host_replay_payload_failed &&
      state_->identity.next_expected_host_payload ==
          state_->plan.value.expected->payloads().host_record_count() &&
      state_->identity.next_expected_replay_input ==
          state_->plan.value.expected->payloads().input_record_count()) {
    return state_->plan.value.expected->payloads().Archive();
  }
  return state_->evidence.host_payload_store.Archive();
}

} // namespace rund::node
