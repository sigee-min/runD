#include "internal.hpp"

namespace rund::compute::detail::residency {

void DirectRecurrenceOwner::quarantine_direct_recurrence() noexcept {
  if (authority_.view_commit_quarantined_locked()) {
    return;
  }
  if (authority_.execution_state_.slot.registration_state != nullptr) {
    authority_.execution_state_.slot.registration_state->set(
        registration_detail::Lifecycle::Quarantined);
  }
  if (authority_.execution_state_.slot.direct_recurrence_admitted) {
    authority_.execution_state_.slot.unknown = true;
    authority_.execution_state_.slot.failed = true;
  }
}

bool DirectRecurrenceOwner::commit_direct_frames(
    const bool success, const bool may_write) noexcept {
  if (authority_.view_commit_quarantined_locked()) {
    return false;
  }
  for (std::size_t index = 0u;
       index < authority_.execution_state_.slot.undo_count; ++index) {
    if (authority_.execution_state_.slot.undo_frames[index] >=
        authority_.frames_.size()) {
      return false;
    }
  }
  for (std::size_t index = 0u;
       index < authority_.execution_state_.slot.undo_count; ++index) {
    Authority::Frame restored = authority_.execution_state_.slot.undo[index];
    const bool writes =
        index < authority_.execution_state_.slot.service_binding_count[0u] &&
        authority_.execution_state_.slot.service_bindings[0u][index].access ==
            Access::Write;
    if (success && writes) {
      restored.state = FrameState::Resident;
      restored.dirty = {};
    } else if (!success && may_write && writes) {
      restored.state = FrameState::Empty;
      restored.dirty = {};
      restored.next_use = 0u;
      restored.retain_until = NeverUse;
    }
    authority_.frames_[authority_.execution_state_.slot.undo_frames[index]] =
        restored;
  }
  return true;
}

DirectAbort DirectRecurrenceOwner::abort_direct_recurrence(
    const DirectRecurrenceLease &lease, const Status failure,
    const execution::TerminalKind terminal, const bool may_write) noexcept {
  static_cast<void>(failure);
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked() ||
      !authority_.execution_state_.slot.direct_recurrence_admitted) {
    return DirectAbort::Invalid;
  }
  if (!lease || authority_.execution_state_.slot.window_final ||
      authority_.execution_state_.slot.registration_state == nullptr ||
      lease.registration_state_ !=
          authority_.execution_state_.slot.registration_state ||
      authority_.execution_state_.slot.registration_nonce == 0u ||
      authority_.execution_state_.slot.registration_nonce !=
          authority_.execution_state_.slot.registration_state->nonce() ||
      lease.registration_nonce_ !=
          authority_.execution_state_.slot.registration_nonce ||
      authority_.execution_state_.slot.registration_state->phase() !=
          registration_detail::Lifecycle::Admitted ||
      authority_.execution_state_.slot.direct_proof_owner !=
          lease.proof_owner_ ||
      authority_.execution_state_.slot.next_sequence != lease.owner_ ||
      authority_.execution_state_.slot.token != lease.token_ ||
      authority_.execution_state_.slot.generation != lease.generation_ ||
      authority_.execution_state_.slot.native_inflight != lease.proof_hi_ ||
      authority_.execution_state_.slot.plan != lease.proof_lo_ ||
      authority_.execution_state_.slot.epochs != lease.iterations_ ||
      authority_.execution_state_.slot.direct_proof_owner == nullptr ||
      !authority_.execution_state_.slot.registration_state->owner_matches(
          lease.proof_owner_)) {
    return DirectAbort::Busy;
  }
  if (terminal == execution::TerminalKind::UnknownMayWrite) {
    quarantine_direct_recurrence();
    return DirectAbort::Quarantined;
  }
  if (!commit_direct_frames(false, may_write)) {
    quarantine_direct_recurrence();
    return DirectAbort::Quarantined;
  }
  authority_.execution_state_.slot.registration_state->set(
      registration_detail::Lifecycle::Active);
  authority_.execution_state_.slot = registry_model::ExecutionSlot{};
  return DirectAbort::Closed;
}

DirectAbort DirectRecurrenceOwner::abort_direct_recurrence_frozen(
    DirectRecurrenceFinal &&prepared) noexcept {
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked() ||
      !authority_.execution_state_.slot.direct_recurrence_admitted) {
    return DirectAbort::Invalid;
  }
  if (!prepared || !authority_.execution_state_.slot.window_final ||
      authority_.execution_state_.slot.registration_state == nullptr ||
      authority_.execution_state_.slot.registration_state->phase() !=
          registration_detail::Lifecycle::Frozen ||
      prepared.registration_state_ !=
          authority_.execution_state_.slot.registration_state ||
      authority_.execution_state_.slot.registration_nonce == 0u ||
      prepared.registration_nonce_ !=
          authority_.execution_state_.slot.registration_nonce ||
      authority_.execution_state_.slot.native_inflight != prepared.proof_hi_ ||
      authority_.execution_state_.slot.plan != prepared.proof_lo_ ||
      authority_.execution_state_.slot.token != prepared.token_ ||
      authority_.execution_state_.slot.generation != prepared.generation_ ||
      authority_.execution_state_.slot.epochs != prepared.iterations_ ||
      authority_.execution_state_.slot.next_sequence != prepared.owner_ ||
      authority_.execution_state_.slot.direct_proof_owner == nullptr ||
      !authority_.execution_state_.slot.registration_state->owner_matches(
          authority_.execution_state_.slot.direct_proof_owner)) {
    return DirectAbort::Busy;
  }
  if (prepared.terminal_ == execution::TerminalKind::UnknownMayWrite) {
    quarantine_direct_recurrence();
    return DirectAbort::Quarantined;
  }
  if (!commit_direct_frames(false, prepared.may_write_)) {
    quarantine_direct_recurrence();
    return DirectAbort::Quarantined;
  }
  authority_.execution_state_.slot.registration_state->set(
      registration_detail::Lifecycle::Active);
  prepared.clear();
  authority_.execution_state_.slot = registry_model::ExecutionSlot{};
  return DirectAbort::Closed;
}

} // namespace rund::compute::detail::residency
