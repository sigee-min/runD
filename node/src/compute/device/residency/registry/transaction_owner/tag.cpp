#include "local.hpp"

#include "../lease_state.hpp"

#include <algorithm>
#include <limits>
#include <mutex>

namespace rund::compute::detail::residency {

bool VirtualTransactionOwner::tag_virtual_transaction_output(
    const std::uint64_t lease_token, const std::uint32_t frame,
    const CacheKey key, const FrameTier tier,
    const std::uint64_t owner_generation) const noexcept {
  if (lease_token == 0u || owner_generation == 0u ||
      frame == std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked()) {
    return false;
  }
  const auto lease =
      std::find_if(authority_.cycle_state_.epochs.begin(),
                   authority_.cycle_state_.epochs.end(),
                   [lease_token](const Authority::LeaseSlot &slot) {
                     return slot.token == lease_token &&
                            (slot.state == Authority::LeaseState::Prepared ||
                             slot.state == Authority::LeaseState::Computing);
                   });
  if (lease == authority_.cycle_state_.epochs.end() ||
      frame >= authority_.frames_.size()) {
    return false;
  }
  const auto binding =
      std::find_if(lease->bindings.begin(), lease->bindings.end(),
                   [frame, key](const CacheBinding &candidate) {
                     return candidate.frame == frame && candidate.key == key &&
                            writes(candidate.access);
                   });
  if (binding == lease->bindings.end()) {
    return false;
  }
  Authority::Frame &value = authority_.frames_[frame];
  if (!value.assigned || value.key != key || value.tier != tier ||
      value.role != FrameRole::Output || value.alias_claims != 0u ||
      (value.state != FrameState::Mapping &&
       value.state != FrameState::Pinned && value.state != FrameState::Dirty) ||
      (value.transaction_generation != 0u &&
       value.transaction_generation != owner_generation)) {
    return false;
  }
  value.transaction_generation = owner_generation;
  return true;
}

} // namespace rund::compute::detail::residency
