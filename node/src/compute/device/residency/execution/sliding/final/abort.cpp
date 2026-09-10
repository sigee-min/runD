#include "abort.hpp"

namespace rund::compute::detail::residency {

void SlidingOwner::quarantine_sliding_locked(execution::Sliding &sliding,
                                             const Status failure) noexcept {
  execution::Sliding::State &state = *sliding.state_;
  authority_.execution_state_.slot.sliding_final_phase =
      SlidingFinalPhase::Quarantined;
  state.mark_unknown(failure);
  state.finalizing = false;
  state.final_nonce = 0u;
  authority_.execution_state_.slot.unknown = true;
  authority_.execution_state_.slot.failed = true;
}

bool SlidingOwner::sliding_final_matches(
    const execution::Plan &plan, const execution::Sliding &sliding,
    const execution::SlidingFinal &final, const ExecutionSlidingFinal &prepared,
    const SlidingFinalPhase phase) const noexcept {
  if (sliding.state_ == nullptr)
    return false;
  const execution::Sliding::State &state = *sliding.state_;
  return authority_.execution_state_.slot.sliding_final_phase == phase &&
         authority_.execution_state_.slot.sliding_admitted &&
         authority_.execution_state_.slot.window_final &&
         prepared.plan_ == authority_.execution_state_.slot.plan &&
         prepared.token_ == authority_.execution_state_.slot.token &&
         prepared.generation_ == authority_.execution_state_.slot.generation &&
         prepared.owner_ == authority_.execution_state_.slot.native_inflight &&
         prepared.sliding_nonce_ != 0u &&
         prepared.nonce_ == authority_.execution_state_.slot.next_sequence &&
         plan.identity() == authority_.execution_state_.slot.plan &&
         plan.epoch_count() == authority_.execution_state_.slot.epochs &&
         !state.closed && !state.quarantine && state.authority_bound &&
         state.finalizing && final.nonce_ == state.final_nonce &&
         final.evidence_.plan == state.plan &&
         final.evidence_.token == state.token &&
         final.evidence_.generation == state.generation &&
         final.evidence_.owner == state.owner &&
         prepared.plan_ == state.plan.lo && prepared.token_ == state.token &&
         prepared.generation_ == state.generation &&
         prepared.owner_ == state.owner &&
         prepared.sliding_nonce_ == state.final_nonce &&
         prepared.success_ == !state.has_failure &&
         final.evidence_.terminal != execution::TerminalKind::UnknownMayWrite &&
         !final.evidence_.quarantined &&
         static_cast<bool>(final.evidence_.status) == prepared.success_ &&
         final.evidence_.has_failure == !prepared.success_;
}

bool SlidingOwner::close_sliding_locked(execution::Sliding &sliding,
                                        const Status failure) noexcept {
  if (authority_.view_commit_quarantined_locked())
    return false;
  execution::Sliding::State &state = *sliding.state_;
  if (!commit_sliding_frames(nullptr, false))
    return false;
  state.inputs = {};
  state.outputs = {};
  state.native_cells = {};
  state.terminals = {};
  state.has_failure = true;
  state.status = failure ? Status::fail(Reason::CompletionInvalid) : failure;
  state.first_failure_terminal = execution::TerminalKind::Known;
  state.terminal = execution::TerminalKind::Known;
  state.first_failure_may_write = false;
  state.closed = true;
  state.authority_bound = false;
  state.finalizing = false;
  state.final_nonce = 0u;
  authority_.execution_state_.slot = registry_model::ExecutionSlot{};
  return true;
}

FinalAbort SlidingOwner::abort_execution_sliding(
    const execution::Plan &plan, execution::Sliding &sliding,
    const execution::SlidingFinal &final, ExecutionSlidingFinal *const prepared,
    const Status failure) noexcept {
  if (sliding.state_ == nullptr || !final)
    return FinalAbort::Invalid;
  std::lock_guard authority_lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked())
    return FinalAbort::Quarantined;
  if (authority_.execution_state_.slot.sliding_final_phase !=
      SlidingFinalPhase::None)
    return FinalAbort::Invalid;
  execution::Sliding::State &state = *sliding.state_;
  std::lock_guard sliding_lock{state.gate};
  const bool base =
      authority_.execution_state_.slot.sliding_admitted &&
      authority_.execution_state_.slot.token != 0u &&
      authority_.execution_state_.slot.generation != 0u &&
      authority_.execution_state_.slot.native_inflight == state.owner &&
      authority_.execution_state_.slot.plan == plan.identity() &&
      authority_.execution_state_.slot.epochs == plan.epoch_count() &&
      state.authority_bound && !state.closed && state.finalizing &&
      final.nonce_ == state.final_nonce && final.evidence_.plan == state.plan &&
      final.evidence_.token == state.token &&
      final.evidence_.generation == state.generation &&
      final.evidence_.owner == state.owner &&
      state.plan.lo == authority_.execution_state_.slot.plan &&
      state.token == authority_.execution_state_.slot.token &&
      state.generation == authority_.execution_state_.slot.generation;
  if (!base) {
    const bool related =
        authority_.execution_state_.slot.sliding_admitted &&
        state.authority_bound &&
        state.owner == authority_.execution_state_.slot.native_inflight &&
        state.plan.lo == authority_.execution_state_.slot.plan &&
        state.token == authority_.execution_state_.slot.token &&
        state.generation == authority_.execution_state_.slot.generation;
    if (related || authority_.execution_state_.slot.window_final ||
        state.finalizing) {
      quarantine_sliding_locked(sliding, failure);
      return FinalAbort::Quarantined;
    }
    return FinalAbort::Invalid;
  }
  if (prepared == nullptr) {
    if (authority_.execution_state_.slot.window_final) {
      quarantine_sliding_locked(sliding, failure);
      return FinalAbort::Quarantined;
    }
  } else if (!*prepared || !authority_.execution_state_.slot.window_final ||
             prepared->plan_ != authority_.execution_state_.slot.plan ||
             prepared->token_ != authority_.execution_state_.slot.token ||
             prepared->generation_ !=
                 authority_.execution_state_.slot.generation ||
             prepared->owner_ !=
                 authority_.execution_state_.slot.native_inflight ||
             prepared->sliding_nonce_ != state.final_nonce ||
             prepared->nonce_ !=
                 authority_.execution_state_.slot.next_sequence ||
             prepared->success_ != !state.has_failure) {
    quarantine_sliding_locked(sliding, failure);
    return FinalAbort::Quarantined;
  }
  if (state.quarantine ||
      final.evidence_.terminal == execution::TerminalKind::UnknownMayWrite ||
      final.evidence_.quarantined) {
    quarantine_sliding_locked(sliding, failure);
    return FinalAbort::Quarantined;
  }
  if (failure.reason() == Reason::PipelinePoisoned) {
    quarantine_sliding_locked(sliding, failure);
    return FinalAbort::Quarantined;
  }
  if (!close_sliding_locked(sliding, failure)) {
    quarantine_sliding_locked(sliding, failure);
    return FinalAbort::Quarantined;
  }
  if (prepared != nullptr)
    *prepared = {};
  return FinalAbort::Closed;
}

FinalAbort
SlidingOwner::close_execution_sliding(const execution::Plan &plan,
                                      execution::Sliding &sliding,
                                      const Status failure) noexcept {
  if (sliding.state_ == nullptr) {
    std::lock_guard authority_lock{authority_.gate_};
    if (authority_.view_commit_quarantined_locked())
      return FinalAbort::Quarantined;
    return authority_.execution_state_.slot.token == 0u &&
                   !authority_.execution_state_.slot.sliding_admitted
               ? FinalAbort::Closed
               : FinalAbort::Invalid;
  }
  std::lock_guard authority_lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked())
    return FinalAbort::Quarantined;
  if (authority_.execution_state_.slot.sliding_final_phase !=
      SlidingFinalPhase::None)
    return FinalAbort::Invalid;
  execution::Sliding::State &state = *sliding.state_;
  std::lock_guard sliding_lock{state.gate};
  if (authority_.execution_state_.slot.token == 0u && state.closed &&
      !state.authority_bound)
    return FinalAbort::Closed;
  const bool no_owner =
      authority_.execution_state_.slot.token == 0u &&
      !authority_.execution_state_.slot.sliding_admitted &&
      !state.authority_bound && state.token == 0u && state.owner == 0u &&
      !state.finalizing && !state.quarantine &&
      !authority_.execution_state_.slot.window_final && state.no_issued() &&
      state.callbacks_quiesced() &&
      authority_.execution_state_.slot.progress.input_services == 0u &&
      authority_.execution_state_.slot.progress.native_dispatches == 0u &&
      authority_.execution_state_.slot.progress.native_completions == 0u &&
      authority_.execution_state_.slot.progress.output_services == 0u &&
      authority_.execution_state_.slot.release_count == 0u &&
      authority_.execution_state_.slot.window_accept_count == 0u &&
      !authority_.execution_state_.slot.native_accepted &&
      !authority_.execution_state_.slot.native_rejected &&
      !authority_.execution_state_.slot.failed &&
      !authority_.execution_state_.slot.unknown &&
      !authority_.execution_state_.slot.cache_admitted &&
      !authority_.execution_state_.slot.window_cache_admitted &&
      !authority_.execution_state_.slot.direct_recurrence_admitted &&
      !authority_.execution_state_.slot.output_admitted;
  if (no_owner) {
    state.closed = true;
    state.final_nonce = 0u;
    return FinalAbort::Closed;
  }
  const bool related =
      authority_.execution_state_.slot.sliding_admitted &&
      authority_.execution_state_.slot.token != 0u &&
      authority_.execution_state_.slot.generation != 0u &&
      authority_.execution_state_.slot.native_inflight != 0u &&
      plan.identity() != 0u &&
      authority_.execution_state_.slot.plan == plan.identity() &&
      authority_.execution_state_.slot.epochs == plan.epoch_count() &&
      state.authority_bound &&
      state.plan == Identity{.lo = authority_.execution_state_.slot.plan} &&
      state.token == authority_.execution_state_.slot.token &&
      state.generation == authority_.execution_state_.slot.generation &&
      state.owner == authority_.execution_state_.slot.native_inflight;
  if (!related) {
    if (authority_.execution_state_.slot.token != 0u ||
        authority_.execution_state_.slot.sliding_admitted ||
        state.authority_bound || state.token != 0u || state.owner != 0u) {
      quarantine_sliding_locked(sliding, failure);
      return FinalAbort::Quarantined;
    }
    return FinalAbort::Invalid;
  }
  const bool live =
      state.closed || state.quarantine || state.finalizing ||
      authority_.execution_state_.slot.window_final ||
      authority_.execution_state_.slot.native_accepted ||
      authority_.execution_state_.slot.native_rejected ||
      authority_.execution_state_.slot.release_count != 0u ||
      authority_.execution_state_.slot.window_accept_count != 0u ||
      !state.no_issued() || !state.callbacks_quiesced() ||
      authority_.execution_state_.slot.progress.input_services != 0u ||
      authority_.execution_state_.slot.progress.native_dispatches != 0u ||
      authority_.execution_state_.slot.progress.native_completions != 0u ||
      authority_.execution_state_.slot.progress.output_services != 0u ||
      authority_.execution_state_.slot.failed ||
      authority_.execution_state_.slot.unknown ||
      authority_.execution_state_.slot.cache_admitted ||
      authority_.execution_state_.slot.window_cache_admitted ||
      authority_.execution_state_.slot.direct_recurrence_admitted ||
      authority_.execution_state_.slot.output_admitted;
  if (live || failure.reason() == Reason::PipelinePoisoned ||
      !authority_.execution_state_.slot.sliding_admitted) {
    quarantine_sliding_locked(sliding, failure);
    return FinalAbort::Quarantined;
  }
  if (!close_sliding_locked(sliding, failure)) {
    quarantine_sliding_locked(sliding, failure);
    return FinalAbort::Quarantined;
  }
  return FinalAbort::Closed;
}

bool SlidingOwner::prepare_execution_sliding(
    const execution::Plan &plan, execution::Sliding &sliding,
    const execution::SlidingFinal &final, ExecutionSlidingFinal &prepared,
    FinalAbort *const abort_result) noexcept {
  if (prepare_execution_sliding(plan, final, prepared))
    return true;
  if (abort_result != nullptr)
    *abort_result = FinalAbort::Invalid;
  if (final) {
    const FinalAbort result = abort_execution_sliding(
        plan, sliding, final, nullptr, Status::fail(Reason::CompletionInvalid));
    if (abort_result != nullptr)
      *abort_result = result;
  }
  return false;
}

} // namespace rund::compute::detail::residency
