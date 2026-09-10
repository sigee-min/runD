#include "internal.hpp"

namespace rund::compute::detail::residency {

bool SlidingOwner::reject_execution_sliding(const execution::Plan &plan,
                                            execution::Sliding &sliding,
                                            const Status failure) noexcept {
  if (sliding.state_ == nullptr || failure) {
    return false;
  }
  std::lock_guard authority_lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked()) {
    return false;
  }
  execution::Sliding::State &state = *sliding.state_;
  std::lock_guard state_lock{state.gate};
  if (!authority_.execution_state_.slot.sliding_admitted ||
      authority_.execution_state_.slot.window_final ||
      authority_.execution_state_.slot.token != state.token ||
      authority_.execution_state_.slot.generation != state.generation ||
      authority_.execution_state_.slot.native_inflight != state.owner ||
      !state.authority_bound || state.model_only || state.closed ||
      state.finalizing || state.has_failure || state.admitted != 0u ||
      !state.no_issued() || plan.identity() == 0u ||
      plan.identity() != authority_.execution_state_.slot.plan ||
      state.invocation.direct_ == nullptr ||
      state.invocation.direct_->identity() != plan.identity()) {
    return false;
  }
  state.fail(execution::SlidingCoordinate{}, failure,
             execution::TerminalKind::Known, false);
  return true;
}

} // namespace rund::compute::detail::residency
