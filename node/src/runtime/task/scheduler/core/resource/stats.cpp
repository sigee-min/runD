#include "../../reactor/ready/set/store.hpp"
#include "../../reactor/registry.hpp"
#include "../../state/storage.hpp"

#include <rund/task/stats/slots.hpp>

#include <algorithm>

namespace rund::node {

void Scheduler::RefreshResourceStats() noexcept {
  auto &metrics = state_->evidence.metrics;
  ::rund::detail::task::Stat(metrics,
                             ::rund::detail::task::StatSlot::ResourceMaxTasks) =
      state_->resources.limits.task_capacity;
  ::rund::detail::task::Stat(
      metrics, ::rund::detail::task::StatSlot::ResourceMaxReactorWaits) =
      state_->resources.limits.reactor_wait_capacity;
  ::rund::detail::task::Stat(
      metrics, ::rund::detail::task::StatSlot::ResourceMaxReadySets) =
      state_->resources.limits.net_ready_set_capacity;
  ::rund::detail::task::Stat(
      metrics, ::rund::detail::task::StatSlot::ResourceMaxReadySetMembers) =
      state_->resources.limits.net_ready_set_member_capacity;
  ::rund::detail::task::Stat(
      metrics,
      ::rund::detail::task::StatSlot::ResourceMaxSocketRegistryEntries) =
      state_->resources.limits.net_socket_registry_capacity;
  const FrameStats frames = state_->resources.frame_arena.stats();
  ::rund::detail::task::Stat(
      metrics, ::rund::detail::task::StatSlot::ResourceCoroutineFrameCapacity) =
      state_->resources.limits.task_capacity;
  ::rund::detail::task::Stat(
      metrics, ::rund::detail::task::StatSlot::ResourceCoroutineFrameBytes) =
      state_->resources.limits.coroutine_frame_bytes;
  ::rund::detail::task::Stat(
      metrics, ::rund::detail::task::StatSlot::ResourceCoroutineFramesLive) =
      frames.live;
  ::rund::detail::task::Stat(
      metrics,
      ::rund::detail::task::StatSlot::ResourceCoroutineFramesHighWater) =
      frames.high_water;
  ::rund::detail::task::Stat(
      metrics,
      ::rund::detail::task::StatSlot::ResourceCoroutineFrameAllocations) =
      frames.allocations;
  ::rund::detail::task::Stat(
      metrics, ::rund::detail::task::StatSlot::ResourceCoroutineFrameReuses) =
      frames.reuses;
  ::rund::detail::task::Stat(
      metrics, ::rund::detail::task::StatSlot::ResourceCoroutineFrameFailures) =
      frames.failures;

  ::rund::detail::task::Stat(
      metrics, ::rund::detail::task::StatSlot::ResourceLiveTasks) =
      state_->resources.live_tasks.load(std::memory_order_relaxed);
  ::rund::detail::task::Stat(
      metrics, ::rund::detail::task::StatSlot::ResourceLiveReactorWaits) =
      static_cast<std::uint64_t>(
          ReactorRegistrySize(state_->reactor.reactor));
  ::rund::detail::task::Stat(
      metrics, ::rund::detail::task::StatSlot::ResourceLiveReadySets) =
      static_cast<std::uint64_t>(
          std::count_if(state_->reactor.reactor_ready_sets.begin(),
                        state_->reactor.reactor_ready_sets.end(),
                        [](const ReactorReadySet &set) { return set.live; }));
  ::rund::detail::task::Stat(
      metrics, ::rund::detail::task::StatSlot::ResourceLiveReadySetMembers) =
      ReactorReadySetMemberCount(state_->reactor.reactor_ready_sets);
  ::rund::detail::task::Stat(
      metrics,
      ::rund::detail::task::StatSlot::ResourceLiveSocketRegistryEntries) = 0u;
  if (state_->reactor.live_net_socket_registry_entries) {
    ::rund::detail::task::Stat(
        metrics,
        ::rund::detail::task::StatSlot::ResourceLiveSocketRegistryEntries) =
        state_->reactor.live_net_socket_registry_entries->load(
            std::memory_order_acquire);
  }
}

} // namespace rund::node
