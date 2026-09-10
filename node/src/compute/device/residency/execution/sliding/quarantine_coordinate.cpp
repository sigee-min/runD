#include "internal.hpp"

namespace rund::compute::detail::residency {

bool SlidingOwner::quarantine_execution_sliding_coordinate(
    const execution::Plan &plan, execution::Sliding &sliding,
    const std::uint64_t coordinate, const Status failure) noexcept {
  if (sliding.state_ == nullptr || failure) {
    return false;
  }
  std::lock_guard authority_lock{authority_.gate_};
  execution::Sliding::State &state = *sliding.state_;
  std::lock_guard state_lock{state.gate};
  const bool identity =
      authority_.execution_state_.slot.sliding_admitted &&
      !authority_.execution_state_.slot.window_final &&
      (authority_.execution_state_.slot.sliding_final_phase ==
           SlidingFinalPhase::None ||
       authority_.execution_state_.slot.sliding_final_phase ==
           SlidingFinalPhase::Quarantined) &&
      authority_.execution_state_.slot.token != 0u &&
      authority_.execution_state_.slot.generation != 0u &&
      authority_.execution_state_.slot.owner_nonce != 0u &&
      authority_.execution_state_.slot.native_inflight != 0u &&
      state.authority_bound && !state.closed &&
      state.owner == authority_.execution_state_.slot.native_inflight &&
      state.plan.lo == authority_.execution_state_.slot.plan &&
      state.token == authority_.execution_state_.slot.token &&
      state.generation == authority_.execution_state_.slot.generation &&
      plan.identity() != 0u &&
      plan.identity() == authority_.execution_state_.slot.plan &&
      plan.epoch_count() == authority_.execution_state_.slot.epochs &&
      state.invocation.direct_ != nullptr &&
      state.invocation.direct_->identity() == plan.identity();
  if (!identity) {
    return false;
  }
  if (state.quarantine) {
    if (!state.has_failure || state.first_failure.ordinal != coordinate ||
        state.first_failure_terminal !=
            execution::TerminalKind::UnknownMayWrite ||
        state.terminal != execution::TerminalKind::UnknownMayWrite ||
        !state.first_failure_may_write || state.status ||
        state.status.reason() != failure.reason()) {
      return false;
    }
    authority_.execution_state_.slot.failed = true;
    authority_.execution_state_.slot.unknown = true;
    authority_.execution_state_.slot.sliding_final_phase =
        SlidingFinalPhase::Quarantined;
    return true;
  }
  if (state.model_only || state.finalizing || coordinate != state.admitted ||
      coordinate >= state.planned || !state.no_issued() ||
      state.terminal_frontier != state.admitted ||
      state.persist_frontier != state.persist_issued) {
    return false;
  }
  std::array<PageUse, execution::UseCapacity> uses{};
  execution::SlidingProjection projection{};
  if (!state.invocation.project(coordinate, uses, projection) ||
      projection.coordinate.ordinal != coordinate) {
    return false;
  }
  if (state.has_failure) {
    state.mark_unknown(failure);
  } else {
    state.fail(projection.coordinate, failure,
               execution::TerminalKind::UnknownMayWrite, true);
  }
  authority_.execution_state_.slot.failed = true;
  authority_.execution_state_.slot.unknown = true;
  authority_.execution_state_.slot.sliding_final_phase =
      SlidingFinalPhase::Quarantined;
  return true;
}

} // namespace rund::compute::detail::residency
