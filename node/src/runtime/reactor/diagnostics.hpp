#pragma once

// Reactor backend lifecycle and scheduler-policy diagnostics. These counters
// are observations only; they never define readiness or wake order.

#include <cstddef>
#include <cstdint>

namespace rund::node {

struct ReactorBackendStats {
  std::uint64_t open_calls = 0u;
  std::uint64_t close_calls = 0u;
  std::uint64_t add_calls = 0u;
  std::uint64_t modify_calls = 0u;
  std::uint64_t remove_calls = 0u;
  std::uint64_t poll_calls = 0u;
  std::uint64_t registration_apply_calls = 0u;
  std::uint64_t max_registration_changes_per_apply = 0u;
  std::uint64_t current_open_handles = 0u;
  std::uint64_t max_open_handles = 0u;
  std::uint64_t current_registered_fds = 0u;
  std::uint64_t max_registered_fds = 0u;
  std::uint64_t max_ready_batch = 0u;
  std::uint64_t ready_expansion_scan_steps = 0u;
  std::uint64_t deferred_remove_marks = 0u;
  std::uint64_t deferred_remove_cancellations = 0u;
  std::uint64_t deferred_remove_flushes = 0u;
  std::uint64_t deferred_remove_invalid_ignored = 0u;
  std::uint64_t deferred_remove_max_pending = 0u;
  std::uint64_t ready_budget_deferrals = 0u;
  std::uint64_t scratch_ready_reuses = 0u;
  std::uint64_t scratch_host_event_reuses = 0u;
  std::uint64_t registration_apply_deferrals = 0u;
  std::uint64_t registration_apply_deferred_flushes = 0u;
  std::uint64_t registration_apply_forced_flushes = 0u;
  std::uint64_t ready_backlog_pushes = 0u;
  std::uint64_t ready_backlog_drains = 0u;
  std::uint64_t ready_backlog_max = 0u;
  std::uint64_t ready_backlog_invalidations = 0u;
  std::uint64_t ready_backlog_scan_steps_avoided = 0u;
};

[[nodiscard]] ReactorBackendStats ReactorBackendStatsSnapshot() noexcept;
void ResetReactorBackendStats() noexcept;
void RecordReactorPlatformOpen() noexcept;
void RecordReactorPlatformClose() noexcept;
void RecordReactorPlatformAdd() noexcept;
void RecordReactorPlatformModify() noexcept;
void RecordReactorPlatformRemove() noexcept;
void RecordReactorPlatformPoll() noexcept;
void RecordReactorRegistrationApplyBatch(std::size_t change_count) noexcept;
void RecordReactorReadyBatch(std::size_t ready_count) noexcept;
void RecordReactorReadyExpansionScanStep(std::size_t count) noexcept;
void RecordReactorDeferredRemoveMark() noexcept;
void RecordReactorDeferredRemoveCancellation() noexcept;
void RecordReactorDeferredRemoveFlush() noexcept;
void RecordReactorDeferredRemoveInvalidIgnored() noexcept;
void RecordReactorDeferredRemovePending(std::size_t pending) noexcept;
void RecordReactorReadyBudgetDeferral(std::size_t deferred) noexcept;
void RecordReactorScratchReadyReuse() noexcept;
void RecordReactorScratchHostEventReuse() noexcept;
void RecordReactorRegistrationApplyDeferral() noexcept;
void RecordReactorRegistrationApplyDeferredFlush() noexcept;
void RecordReactorRegistrationApplyForcedFlush() noexcept;
void RecordReactorReadyBacklogPush(std::size_t count) noexcept;
void RecordReactorReadyBacklogDrain(std::size_t count) noexcept;
void RecordReactorReadyBacklogInvalidation(std::size_t count) noexcept;
void RecordReactorReadyBacklogScanStepsAvoided(std::size_t count) noexcept;

} // namespace rund::node
