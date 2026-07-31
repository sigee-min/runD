#include <algorithm>

#include "../../../../reactor/diagnostics.hpp"

namespace rund::node {
namespace {

ReactorBackendStats g_reactor_stats{};

}  // namespace

ReactorBackendStats ReactorBackendStatsSnapshot() noexcept {
  return g_reactor_stats;
}

void ResetReactorBackendStats() noexcept {
  g_reactor_stats = ReactorBackendStats{};
}

void RecordReactorPlatformOpen() noexcept {
  ++g_reactor_stats.open_calls;
  ++g_reactor_stats.current_open_handles;
  g_reactor_stats.max_open_handles = std::max(
      g_reactor_stats.max_open_handles, g_reactor_stats.current_open_handles);
}

void RecordReactorPlatformClose() noexcept {
  ++g_reactor_stats.close_calls;
  if (g_reactor_stats.current_open_handles != 0u) {
    --g_reactor_stats.current_open_handles;
  }
  g_reactor_stats.current_registered_fds = 0u;
}

void RecordReactorPlatformAdd() noexcept {
  ++g_reactor_stats.add_calls;
  ++g_reactor_stats.current_registered_fds;
  g_reactor_stats.max_registered_fds =
      std::max(g_reactor_stats.max_registered_fds,
               g_reactor_stats.current_registered_fds);
}

void RecordReactorPlatformModify() noexcept { ++g_reactor_stats.modify_calls; }

void RecordReactorPlatformRemove() noexcept {
  ++g_reactor_stats.remove_calls;
  if (g_reactor_stats.current_registered_fds != 0u) {
    --g_reactor_stats.current_registered_fds;
  }
}

void RecordReactorPlatformPoll() noexcept { ++g_reactor_stats.poll_calls; }

void RecordReactorRegistrationApplyBatch(
    const std::size_t change_count) noexcept {
  ++g_reactor_stats.registration_apply_calls;
  g_reactor_stats.max_registration_changes_per_apply = std::max<std::uint64_t>(
      g_reactor_stats.max_registration_changes_per_apply, change_count);
}

void RecordReactorReadyBatch(const std::size_t ready_count) noexcept {
  g_reactor_stats.max_ready_batch =
      std::max<std::uint64_t>(g_reactor_stats.max_ready_batch, ready_count);
}

void RecordReactorReadyExpansionScanStep(const std::size_t count) noexcept {
  g_reactor_stats.ready_expansion_scan_steps += count;
}

void RecordReactorDeferredRemoveMark() noexcept {
  ++g_reactor_stats.deferred_remove_marks;
}

void RecordReactorDeferredRemoveCancellation() noexcept {
  ++g_reactor_stats.deferred_remove_cancellations;
}

void RecordReactorDeferredRemoveFlush() noexcept {
  ++g_reactor_stats.deferred_remove_flushes;
}

void RecordReactorDeferredRemoveInvalidIgnored() noexcept {
  ++g_reactor_stats.deferred_remove_invalid_ignored;
}

void RecordReactorDeferredRemovePending(const std::size_t pending) noexcept {
  g_reactor_stats.deferred_remove_max_pending = std::max<std::uint64_t>(
      g_reactor_stats.deferred_remove_max_pending, pending);
}

void RecordReactorReadyBudgetDeferral(const std::size_t deferred) noexcept {
  g_reactor_stats.ready_budget_deferrals += deferred;
}

void RecordReactorScratchReadyReuse() noexcept {
  ++g_reactor_stats.scratch_ready_reuses;
}

void RecordReactorScratchHostEventReuse() noexcept {
  ++g_reactor_stats.scratch_host_event_reuses;
}

void RecordReactorRegistrationApplyDeferral() noexcept {
  ++g_reactor_stats.registration_apply_deferrals;
}

void RecordReactorRegistrationApplyDeferredFlush() noexcept {
  ++g_reactor_stats.registration_apply_deferred_flushes;
}

void RecordReactorRegistrationApplyForcedFlush() noexcept {
  ++g_reactor_stats.registration_apply_forced_flushes;
}

void RecordReactorReadyBacklogPush(const std::size_t count) noexcept {
  g_reactor_stats.ready_backlog_pushes += count;
  g_reactor_stats.ready_backlog_max =
      std::max<std::uint64_t>(g_reactor_stats.ready_backlog_max, count);
}

void RecordReactorReadyBacklogDrain(const std::size_t count) noexcept {
  g_reactor_stats.ready_backlog_drains += count;
}

void RecordReactorReadyBacklogInvalidation(const std::size_t count) noexcept {
  g_reactor_stats.ready_backlog_invalidations += count;
}

void RecordReactorReadyBacklogScanStepsAvoided(
    const std::size_t count) noexcept {
  g_reactor_stats.ready_backlog_scan_steps_avoided += count;
}

}  // namespace rund::node
