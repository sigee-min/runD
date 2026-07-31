#include <rund/session/config.hpp>
#include <rund/task/stats/slots.hpp>

#include "../../../../host/net/registry/access.hpp"
#include "../../../reactor/platform.hpp"
#include "../access.hpp"
#include "../reactor/registry.hpp"
#include "../state/model/join.hpp"
#include "../state/model/task.hpp"
#include "../state/model/timer.hpp"
#include "../state/segment.hpp"
#include "../state/storage.hpp"

#include <utility>

namespace rund::node {

namespace {

std::atomic<std::uint32_t> &ExternalSocketRegistryEntries() noexcept {
  static std::atomic<std::uint32_t> entries{0u};
  return entries;
}

bool ReserveExternalSocketRegistry() noexcept {
  auto &entries = ExternalSocketRegistryEntries();
  std::uint32_t current = entries.load(std::memory_order_acquire);
  constexpr std::uint32_t capacity =
      ::rund::SchedulerConfig{}.net_socket_registry_capacity;
  while (current < capacity) {
    if (entries.compare_exchange_weak(current, current + 1u,
                                      std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
      return true;
    }
  }
  return false;
}

void ReleaseExternalSocketRegistry() noexcept {
  auto &entries = ExternalSocketRegistryEntries();
  std::uint32_t current = entries.load(std::memory_order_acquire);
  while (current != 0u) {
    if (entries.compare_exchange_weak(current, current - 1u,
                                      std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
      return;
    }
  }
}

void ReleaseSocketRegistryOwner(
    const ::rund::net::SocketRegistryOwner owner) noexcept {
  if (owner.external()) {
    ReleaseExternalSocketRegistry();
    return;
  }
  const std::shared_ptr<std::atomic<std::uint32_t>> live_entries =
      owner.live_entries;
  if (!live_entries) {
    return;
  }
  std::uint32_t current = live_entries->load(std::memory_order_acquire);
  while (current != 0u) {
    if (live_entries->compare_exchange_weak(current, current - 1u,
                                            std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
      return;
    }
  }
}

} // namespace

thread_local Scheduler *Scheduler::active_ = nullptr;
std::atomic<std::uint64_t> Scheduler::next_scheduler_id_{1u};

Scheduler::Scheduler() : state_(new SchedulerState{}) {
  ::rund::detail::task::Stat(state_->evidence.metrics,
                             ::rund::detail::task::StatSlot::TaskWorkers) = 1u;
  ::rund::detail::task::Stat(state_->evidence.metrics,
                             ::rund::detail::task::StatSlot::TraceHash) =
      ::rund::detail::task::kTraceHashSeed;
  state_->owner = this;
  state_->identity.scheduler_id =
      next_scheduler_id_.fetch_add(1u, std::memory_order_relaxed);
  state_->reactor.live_net_socket_registry_entries =
      std::make_shared<std::atomic<std::uint32_t>>(0u);
}

Scheduler::~Scheduler() {
  Reset();
  delete state_;
}

void Scheduler::SetActive(Scheduler *const scheduler) noexcept {
  active_ = scheduler;
  ::rund::detail::task::frame::Bind(
      scheduler == nullptr ? nullptr
                           : &scheduler->state_->resources.frame_arena);
}

task::Handle Scheduler::InvalidHandle(const ReasonCode code) noexcept {
  return ::rund::detail::task::HandleAccess::Fail(code);
}

Scheduler *scheduler_access::ActiveScheduler() noexcept {
  return Scheduler::Active();
}

std::uint32_t
scheduler_access::CoroutineFrameByteLimit(const Scheduler &scheduler) noexcept {
  return scheduler.state_->resources.limits.coroutine_frame_bytes;
}

::rund::net::SocketRegistryOwner
Scheduler::ActiveNetworkSocketRegistryOwner() const noexcept {
  if (state_ == nullptr || state_->identity.scheduler_id == 0u) {
    return {};
  }
  return ::rund::net::SocketRegistryOwner{
      .scheduler_id = state_->identity.scheduler_id,
      .live_entries = state_->reactor.live_net_socket_registry_entries,
  };
}

bool Scheduler::Configure(const ::rund::SchedulerConfig config,
                          const rund::kernel::ParallelRuntimeProvider provider,
                          ::rund::ReplayConfig replay,
                          const ::rund::host::random::RunSeed random_seed) {
  const std::shared_ptr<std::atomic<std::uint32_t>> registry_entries =
      state_->reactor.live_net_socket_registry_entries;
  if (registry_entries != nullptr &&
      registry_entries->load(std::memory_order_acquire) >
          config.net_socket_registry_capacity) {
    state_->lanes.lane_code = ReasonCode::TaskCapacityExceeded;
    return false;
  }
  state_->resources.limits = config;
  state_->resources.limits.host_io_capacity =
      std::min(state_->resources.limits.host_io_capacity,
               state_->resources.limits.task_capacity);
  state_->resources.limits.host_payload_capacity_bytes =
      std::min(state_->resources.limits.host_payload_capacity_bytes,
               replay.storage.max_bytes);
  const std::optional<replay_detail::payload::Limits> replay_limits =
      replay_detail::payload::Limits::runtime(
          state_->resources.limits.host_event_capacity, replay.input_capacity,
          state_->resources.limits.host_payload_capacity_bytes);
  if (!replay_limits.has_value()) {
    state_->lanes.lane_code = ReasonCode::HostReplayStorageInvalid;
    return false;
  }
  state_->evidence.input_capacity = replay.input_capacity;
  state_->lanes.kernel_provider = provider;
  state_->identity.random_seed = random_seed;
  Reset();
  if (state_->resources.limits.host_payload_capacity_bytes >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    state_->lanes.lane_code = ReasonCode::TaskSchedulerAllocationFailed;
    return false;
  }
  try {
    state_->evidence.input_bytes =
        std::make_shared<std::vector<std::byte>>(static_cast<std::size_t>(
            state_->resources.limits.host_payload_capacity_bytes));
  } catch (...) {
    state_->lanes.lane_code = ReasonCode::TaskSchedulerAllocationFailed;
    return false;
  }
  state_->evidence.host_payload_store = replay_detail::payload::Store{
      replay.storage, *replay_limits, replay.diagnostic};
  if (!PrepareHostIo()) {
    state_->lanes.lane_code = ReasonCode::TaskSchedulerAllocationFailed;
    return false;
  }
  if (!state_->plan.configure_handles(
          state_->resources.limits.host_handle_capacity)) {
    state_->lanes.lane_code = ReasonCode::TaskSchedulerAllocationFailed;
    return false;
  }
  state_->ready.records.reserve(state_->resources.limits.task_capacity);
  state_->ready.join_waits.reserve(state_->resources.limits.task_capacity);
  state_->ready.ready.configure(state_->resources.limits.task_capacity);
  const std::size_t index_capacity =
      static_cast<std::size_t>(state_->resources.limits.task_capacity) * 2u +
      1u;
  state_->ready.record_index_slots.assign(index_capacity, 0u);
  state_->ready.record_index_size = 0u;
  state_->ready.record_index_deleted = 0u;
  state_->ready.free_record_head = 0u;
  state_->ready.timers.reserve(state_->resources.limits.timer_capacity);
  state_->ready.timer_wait_id_index.reserve(
      state_->resources.limits.timer_capacity);
  state_->evidence.observations.reserve(
      state_->resources.limits.observation_capacity);
  state_->evidence.host_events.reserve(
      state_->resources.limits.host_event_capacity);
  ReactorRuntime &reactor = state_->reactor.reactor;
  const std::size_t reactor_capacity =
      state_->resources.limits.reactor_wait_capacity;
  if (!ReactorRegistryPrepare(reactor, reactor_capacity)) {
    state_->lanes.lane_code = ReasonCode::TaskSchedulerAllocationFailed;
    return false;
  }
  reactor.changes.reserve(reactor_capacity * 2u);
  reactor.probe_ready.reserve(1u);
  reactor.ready.reserve(reactor_capacity);
  reactor.ready_backlog.reserve(reactor_capacity);
  reactor.ordered_ready_scratch.reserve(reactor_capacity);
  reactor.budget_ready_scratch.reserve(reactor_capacity);
  reactor.drain_ready_scratch.reserve(reactor_capacity);
  reactor.removed_wait_scratch.reserve(reactor_capacity);
  reactor.stale_wait_scratch.reserve(reactor_capacity);
  reactor.previous_interest_scratch.reserve(reactor_capacity);
  state_->reactor.reactor_many_groups.reserve(
      state_->resources.limits.task_capacity);
  state_->reactor.reactor_many_requests.reserve(reactor_capacity);
  state_->reactor.reactor_many_request_scratch.reserve(reactor_capacity);
  state_->reactor.reactor_many_index_scratch.reserve(reactor_capacity);
  state_->reactor.reactor_many_group_id_scratch.reserve(
      state_->resources.limits.task_capacity);
  state_->reactor.reactor_many_event_slots.reserve(reactor_capacity);
  state_->reactor.reactor_many_event_slots_scratch.reserve(reactor_capacity);
  state_->reactor.reactor_many_poll_request_scratch.reserve(reactor_capacity);
  state_->reactor.reactor_many_ready_result_scratch.reserve(reactor_capacity);
  state_->reactor.reactor_ready_code_scratch.reserve(reactor_capacity);
  state_->reactor.reactor_socket_lease_scratch.reserve(reactor_capacity);
  state_->reactor.reactor_ready_sets.reserve(
      state_->resources.limits.net_ready_set_capacity);
  const ReactorPlatformOpResult platform_prepared =
      PrepareReactorPlatform(reactor.platform, reactor_capacity);
  if (!platform_prepared.ok) {
    state_->lanes.lane_code = ReasonCode::ReactorWaitCapacityExceeded;
    return false;
  }
  if (!state_->resources.callable_pool.configure(
          state_->resources.limits.task_capacity)) {
    state_->lanes.lane_code = ReasonCode::TaskCapacityExceeded;
    return false;
  }
  ::rund::detail::task::Stat(state_->evidence.metrics,
                             ::rund::detail::task::StatSlot::TaskWorkers) =
      std::max<std::uint32_t>(1u, state_->resources.limits.task_workers);
  ::rund::detail::task::Stat(state_->evidence.metrics,
                             ::rund::detail::task::StatSlot::TraceHash) =
      ::rund::detail::task::kTraceHashSeed;
  const task::Status frame_status =
      state_->resources.frame_arena.configure(FrameLimits{
          .capacity = state_->resources.limits.task_capacity,
          .bytes = state_->resources.limits.coroutine_frame_bytes,
          .alignment = state_->resources.limits.coroutine_frame_alignment});
  if (!frame_status) {
    state_->lanes.lane_code = frame_status.code();
    return false;
  }
  const task::Status completion_status =
      state_->resources.completion_pool.configure(CompletionLimits{
          .capacity = state_->resources.limits.task_capacity,
          .result_bytes = state_->resources.limits.task_result_bytes,
          .result_alignment = state_->resources.limits.task_result_alignment});
  if (!completion_status) {
    state_->lanes.lane_code = completion_status.code();
    return false;
  }
  if (!StartLanes()) {
    state_->lanes.lane_code = ReasonCode::TaskWorkersInvalid;
    return false;
  }
  return true;
}

bool Scheduler::TryAdmitNetworkSocketRegistry() noexcept {
  (void)TrapLaneOwnedSegmentPrimitive();
  EnsureCurrentCommit();
  std::shared_ptr<std::atomic<std::uint32_t>> live_entries =
      state_->reactor.live_net_socket_registry_entries;
  if (!live_entries) {
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::NetworkAdmissionRejections);
    CompletePrimitiveCommit();
    return false;
  }
  std::uint32_t current = live_entries->load(std::memory_order_acquire);
  while (current < state_->resources.limits.net_socket_registry_capacity) {
    if (live_entries->compare_exchange_weak(current, current + 1u,
                                            std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
      CompletePrimitiveCommit();
      return true;
    }
  }
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::NetworkAdmissionRejections);
  CompletePrimitiveCommit();
  return false;
}

} // namespace rund::node

namespace rund::net {

SocketRegistryOwner SocketRegistryAccess::ActiveOwner() noexcept {
  const node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr) {
    return SocketRegistryOwner{.scheduler_id =
                                   SocketRegistryOwner::external_id};
  }
  return scheduler->ActiveNetworkSocketRegistryOwner();
}

bool SocketRegistryAccess::TryAdmit(const SocketRegistryOwner owner) noexcept {
  if (owner.external()) {
    return node::ReserveExternalSocketRegistry();
  }
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr ||
      scheduler->ActiveNetworkSocketRegistryOwner().scheduler_id !=
          owner.scheduler_id) {
    return false;
  }
  return scheduler->TryAdmitNetworkSocketRegistry();
}

void SocketRegistryAccess::ReleaseOwner(
    const SocketRegistryOwner owner) noexcept {
  node::ReleaseSocketRegistryOwner(owner);
}

} // namespace rund::net
