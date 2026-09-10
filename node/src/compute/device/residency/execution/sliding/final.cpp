#include "internal.hpp"

namespace rund::compute::detail::residency::execution {

bool Sliding::prepare_final(SlidingFinal &final) noexcept {
  final = {};
  if (state_ == nullptr) {
    return false;
  }
  std::lock_guard lock{state_->gate};
  if (!state_->runnable() || !state_->authority_bound || state_->quarantine ||
      !state_->callbacks_quiesced() ||
      (!state_->has_failure &&
       (!state_->no_issued() || state_->terminal_frontier != state_->admitted ||
        state_->persist_frontier != state_->persist_issued ||
        state_->admitted != state_->planned))) {
    return false;
  }
  state_->final_nonce =
      state_->final_nonce == std::numeric_limits<std::uint64_t>::max()
          ? 1u
          : state_->final_nonce + 1u;
  if (state_->final_nonce == 0u) {
    state_->final_nonce = 1u;
  }
  state_->evidence(final.evidence_);
  final.nonce_ = state_->final_nonce;
  state_->finalizing = true;
  return true;
}

bool Sliding::accept_final(const SlidingFinal &final,
                           ExecutionSlidingReceipt &&receipt,
                           SlidingEvidence &evidence) noexcept {
  evidence = {};
  if (state_ == nullptr || !final || !receipt) {
    return false;
  }
  std::lock_guard lock{state_->gate};
  if (state_->closed || !state_->authority_bound || !state_->finalizing ||
      final.nonce_ != state_->final_nonce ||
      final.evidence_.plan != state_->plan ||
      final.evidence_.token != state_->token ||
      final.evidence_.generation != state_->generation ||
      final.evidence_.owner != state_->owner ||
      receipt.plan_ != state_->plan.lo || receipt.token_ != state_->token ||
      receipt.generation_ != state_->generation ||
      receipt.owner_ != state_->owner || receipt.nonce_ == 0u ||
      receipt.sliding_nonce_ != state_->final_nonce ||
      receipt.success_ != !state_->has_failure) {
    return false;
  }
  if (state_->has_failure) {
    state_->inputs = {};
    state_->outputs = {};
    state_->native_cells = {};
    state_->terminals = {};
  }
  evidence = final.evidence_;
  receipt = {};
  state_->closed = true;
  state_->finalizing = false;
  state_->final_nonce = 0u;
  state_->authority_bound = false;
  return true;
}

bool Sliding::close_model(SlidingEvidence &evidence) noexcept {
  evidence = {};
  if (state_ == nullptr) {
    return false;
  }
  std::lock_guard lock{state_->gate};
  // Unknown quarantines already-terminal cells but cannot cancel a different
  // accepted callback.  Every non-quarantined issued cell must quiesce before
  // the caller may release the controller lifetime.
  if (state_->closed || !state_->model_only || state_->authority_bound ||
      !state_->no_issued() ||
      (!state_->quarantine &&
       (state_->terminal_frontier != state_->admitted ||
        state_->persist_frontier != state_->persist_issued ||
        (!state_->has_failure && state_->admitted != state_->planned)))) {
    return false;
  }
  state_->evidence(evidence);
  state_->closed = true;
  return true;
}

bool Sliding::snapshot(SlidingEvidence &evidence) const noexcept {
  evidence = {};
  if (state_ == nullptr) {
    return false;
  }
  std::lock_guard lock{state_->gate};
  state_->evidence(evidence);
  return true;
}

bool Sliding::quarantined() const noexcept {
  if (state_ == nullptr) {
    return false;
  }
  std::lock_guard lock{state_->gate};
  return state_->quarantine;
}

} // namespace rund::compute::detail::residency::execution
