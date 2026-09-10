#include "internal.hpp"

namespace rund::compute::detail::residency {

bool DirectRecurrenceOwner::prepare_direct_recurrence_final(
    const DirectRecurrenceLease &lease, const Status status,
    const execution::TerminalKind terminal, const bool may_write,
    const std::uint64_t completed, DirectRecurrenceFinal &prepared) noexcept {
  prepared.clear();
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked() ||
      !authority_.execution_state_.slot.direct_recurrence_admitted ||
      authority_.execution_state_.slot.registration_state == nullptr) {
    return false;
  }
  const bool authenticated =
      lease && !authority_.execution_state_.slot.window_final &&
      authority_.execution_state_.slot.token != 0u &&
      lease.registration_state_ ==
          authority_.execution_state_.slot.registration_state &&
      authority_.execution_state_.slot.registration_nonce != 0u &&
      authority_.execution_state_.slot.registration_nonce ==
          authority_.execution_state_.slot.registration_state->nonce() &&
      lease.registration_nonce_ ==
          authority_.execution_state_.slot.registration_nonce &&
      authority_.execution_state_.slot.registration_state->phase() ==
          registration_detail::Lifecycle::Admitted &&
      authority_.execution_state_.slot.registration_state->owner_matches(
          lease.proof_owner_) &&
      authority_.execution_state_.slot.direct_proof_owner ==
          lease.proof_owner_ &&
      authority_.execution_state_.slot.native_inflight == lease.proof_hi_ &&
      authority_.execution_state_.slot.plan == lease.proof_lo_ &&
      authority_.execution_state_.slot.token == lease.token_ &&
      authority_.execution_state_.slot.generation == lease.generation_ &&
      authority_.execution_state_.slot.next_sequence == lease.owner_ &&
      authority_.execution_state_.slot.epochs == lease.iterations_;
  if (!authenticated) {
    return false;
  }
  const bool success = static_cast<bool>(status);
  const bool valid_terminal =
      (terminal == execution::TerminalKind::Known &&
       ((success && completed == lease.iterations_ && may_write) ||
        (!success && completed <= lease.iterations_))) ||
      (!success && terminal == execution::TerminalKind::UnknownMayWrite &&
       may_write && completed <= lease.iterations_);
  if (!valid_terminal) {
    return false;
  }

  if (!authority_.execution_state_.slot.registration_state->transition(
          registration_detail::Lifecycle::Admitted,
          registration_detail::Lifecycle::Frozen)) {
    authority_.execution_state_.slot.registration_state->set(
        registration_detail::Lifecycle::Quarantined);
    authority_.execution_state_.slot.unknown = true;
    authority_.execution_state_.slot.failed = true;
    return false;
  }

  authority_.execution_state_.slot.window_final = true;
  prepared.proof_hi_ = lease.proof_hi_;
  prepared.proof_lo_ = lease.proof_lo_;
  prepared.token_ = lease.token_;
  prepared.generation_ = lease.generation_;
  prepared.owner_ = lease.owner_;
  prepared.iterations_ = lease.iterations_;
  prepared.completed_ = completed;
  prepared.status_ = status;
  prepared.terminal_ = terminal;
  prepared.may_write_ = may_write;
  prepared.registration_state_ =
      authority_.execution_state_.slot.registration_state;
  prepared.registration_nonce_ =
      authority_.execution_state_.slot.registration_nonce;
  return true;
}

} // namespace rund::compute::detail::residency
