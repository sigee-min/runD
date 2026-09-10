#include "../internal.hpp"

namespace rund::compute::detail::residency {

bool DirectRecurrenceOwner::direct_lease_matches(
    const DirectRecurrenceLease &lease) const noexcept {
  return lease && authority_.execution_state_.slot.direct_recurrence_admitted &&
         authority_.execution_state_.slot.registration_state != nullptr &&
         lease.registration_state_ ==
             authority_.execution_state_.slot.registration_state &&
         lease.registration_nonce_ ==
             authority_.execution_state_.slot.registration_nonce &&
         authority_.execution_state_.slot.registration_nonce ==
             authority_.execution_state_.slot.registration_state->nonce() &&
         lease.proof_owner_ ==
             authority_.execution_state_.slot.direct_proof_owner &&
         authority_.execution_state_.slot.window_final &&
         authority_.execution_state_.slot.native_inflight == lease.proof_hi_ &&
         authority_.execution_state_.slot.plan == lease.proof_lo_ &&
         authority_.execution_state_.slot.token == lease.token_ &&
         authority_.execution_state_.slot.generation == lease.generation_ &&
         authority_.execution_state_.slot.next_sequence == lease.owner_ &&
         authority_.execution_state_.slot.epochs == lease.iterations_;
}

bool DirectRecurrenceOwner::clean_direct_registration_rows() const noexcept {
  if (authority_.execution_state_.slot.direct_binding_count == 0u ||
      authority_.execution_state_.slot.direct_binding_count >
          authority_.execution_state_.slot.direct_bindings.size()) {
    return false;
  }
  for (std::size_t left = 0u;
       left < authority_.execution_state_.slot.direct_binding_count; ++left) {
    const ResidentRecurrenceBinding &binding =
        authority_.execution_state_.slot.direct_bindings[left];
    const std::size_t index = binding.region.first;
    if (index >= authority_.frames_.size() ||
        !direct_recurrence_detail::exact_view(authority_.frames_[index],
                                              binding) ||
        authority_.frames_[index].direct_registration !=
            authority_.execution_state_.slot.registration_state) {
      return false;
    }
    const Authority::Frame &frame = authority_.frames_[index];
    if (frame.alias_claims != 0u || !frame.dirty.empty() ||
        (frame.state != FrameState::Resident &&
         frame.state != FrameState::Empty)) {
      return false;
    }
    for (std::size_t right = 0u; right < left; ++right) {
      if (authority_.execution_state_.slot.direct_bindings[right]
              .region.first == index) {
        return false;
      }
    }
  }
  return true;
}

} // namespace rund::compute::detail::residency
