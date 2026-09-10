#include "../internal.hpp"

namespace rund::compute::detail::residency {
namespace {

[[nodiscard]] ExecutionClose close_result(const RegistrationResult result,
                                          const bool success = false) noexcept {
  const AuthorityFailure failure =
      result == RegistrationResult::Busy      ? AuthorityFailure::Busy
      : result == RegistrationResult::Invalid ? AuthorityFailure::Invalid
                                              : AuthorityFailure::None;
  return ExecutionClose{.failure = failure,
                        .success = success,
                        .quarantined =
                            result == RegistrationResult::Quarantined,
                        .registration = result};
}

} // namespace

ExecutionClose DirectRecurrenceOwner::stage_direct_recurrence_final(
    DirectRecurrenceFinal &&prepared) noexcept {
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked() || !prepared ||
      !authority_.execution_state_.slot.direct_recurrence_admitted ||
      !authority_.execution_state_.slot.window_final ||
      prepared.registration_state_ == nullptr ||
      prepared.registration_state_ !=
          authority_.execution_state_.slot.registration_state ||
      authority_.execution_state_.slot.registration_nonce == 0u ||
      prepared.registration_nonce_ !=
          authority_.execution_state_.slot.registration_nonce ||
      prepared.registration_nonce_ != prepared.registration_state_->nonce() ||
      authority_.execution_state_.slot.registration_state->phase() !=
          registration_detail::Lifecycle::Frozen ||
      authority_.execution_state_.slot.native_inflight != prepared.proof_hi_ ||
      authority_.execution_state_.slot.plan != prepared.proof_lo_ ||
      authority_.execution_state_.slot.token != prepared.token_ ||
      authority_.execution_state_.slot.generation != prepared.generation_ ||
      authority_.execution_state_.slot.next_sequence != prepared.owner_ ||
      authority_.execution_state_.slot.epochs != prepared.iterations_) {
    return close_result(RegistrationResult::Busy);
  }
  if (prepared.terminal_ == execution::TerminalKind::UnknownMayWrite) {
    quarantine_direct_recurrence();
    prepared.clear();
    return close_result(RegistrationResult::Quarantined);
  }
  const bool success = static_cast<bool>(prepared.status_);
  if (!commit_direct_frames(success, prepared.may_write_)) {
    quarantine_direct_recurrence();
    return close_result(RegistrationResult::Quarantined);
  }
  authority_.execution_state_.slot.direct_success = success;
  authority_.execution_state_.slot.direct_released = false;
  authority_.execution_state_.slot.registration_state->set(
      registration_detail::Lifecycle::Pending);
  prepared.clear();
  return close_result(RegistrationResult::Done, success);
}

RegistrationResult DirectRecurrenceOwner::release_direct_recurrence_pending(
    const DirectRecurrenceLease &lease,
    const std::shared_ptr<const registration_detail::State> &state) noexcept {
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked() || state == nullptr ||
      authority_.execution_state_.slot.registration_state != state ||
      authority_.execution_state_.slot.registration_nonce == 0u ||
      authority_.execution_state_.slot.registration_nonce != state->nonce() ||
      lease.registration_state_ != state ||
      lease.registration_nonce_ != state->nonce()) {
    return RegistrationResult::Busy;
  }
  if (!direct_lease_matches(lease)) {
    return RegistrationResult::Busy;
  }
  const auto phase =
      authority_.execution_state_.slot.registration_state->phase();
  if (phase == registration_detail::Lifecycle::Inflight) {
    return RegistrationResult::Busy;
  }
  if (phase == registration_detail::Lifecycle::Quarantined) {
    return RegistrationResult::Quarantined;
  }
  if (phase != registration_detail::Lifecycle::Pending) {
    return RegistrationResult::Busy;
  }
  if (authority_.execution_state_.slot.registration_state->policy() ==
          registration_detail::Policy::Retain ||
      authority_.execution_state_.slot.direct_released) {
    return RegistrationResult::Done;
  }
  if (!clean_direct_registration_rows()) {
    quarantine_direct_recurrence();
    return RegistrationResult::Quarantined;
  }
  for (std::size_t index = 0u;
       index < authority_.execution_state_.slot.direct_binding_count; ++index) {
    authority_.frames_[authority_.execution_state_.slot.direct_bindings[index]
                           .region.first] = Authority::Frame{};
  }
  authority_.execution_state_.slot.direct_released = true;
  return RegistrationResult::Done;
}

ExecutionClose DirectRecurrenceOwner::finish_direct_recurrence(
    const DirectRecurrenceLease &lease,
    const std::shared_ptr<const registration_detail::State> &state,
    void *const user, const DirectRecurrencePublication publication) noexcept {
  if (user == nullptr || publication == nullptr) {
    return close_result(RegistrationResult::Invalid);
  }

  bool success = false;
  registration_detail::Policy policy = registration_detail::Policy::Release;
  {
    std::unique_lock lock{authority_.gate_};
    if (authority_.view_commit_quarantined_locked() || state == nullptr ||
        authority_.execution_state_.slot.registration_state != state ||
        authority_.execution_state_.slot.registration_nonce != state->nonce() ||
        !direct_lease_matches(lease)) {
      return close_result(RegistrationResult::Busy);
    }
    const auto phase =
        authority_.execution_state_.slot.registration_state->phase();
    if (phase == registration_detail::Lifecycle::Inflight) {
      return close_result(RegistrationResult::Busy);
    }
    if (phase == registration_detail::Lifecycle::Quarantined) {
      return close_result(RegistrationResult::Quarantined);
    }
    if (phase != registration_detail::Lifecycle::Pending ||
        (authority_.execution_state_.slot.registration_state->policy() ==
             registration_detail::Policy::Release &&
         !authority_.execution_state_.slot.direct_released) ||
        (authority_.execution_state_.slot.registration_state->policy() ==
             registration_detail::Policy::Retain &&
         authority_.execution_state_.slot.direct_released)) {
      return close_result(RegistrationResult::Busy);
    }
    success = authority_.execution_state_.slot.direct_success;
    policy = authority_.execution_state_.slot.registration_state->policy();
    if (!authority_.execution_state_.slot.registration_state->transition(
            registration_detail::Lifecycle::Pending,
            registration_detail::Lifecycle::Inflight)) {
      return close_result(RegistrationResult::Busy);
    }
  }

  publication(user, success);

  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked() ||
      !direct_lease_matches(lease) ||
      authority_.execution_state_.slot.registration_state->phase() !=
          registration_detail::Lifecycle::Inflight) {
    quarantine_direct_recurrence();
    return close_result(RegistrationResult::Quarantined);
  }
  authority_.execution_state_.slot.registration_state->set(
      policy == registration_detail::Policy::Release
          ? registration_detail::Lifecycle::Released
          : registration_detail::Lifecycle::Active);
  authority_.execution_state_.slot = registry_model::ExecutionSlot{};
  return close_result(RegistrationResult::Done, success);
}

} // namespace rund::compute::detail::residency
