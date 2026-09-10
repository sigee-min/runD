#include "../release_check.hpp"

#include <algorithm>
#include <array>

namespace rund::compute::detail::residency::release_detail {

bool ReleaseCheck::cycle(const Authority &authority,
                         const std::span<const FrameRegion>,
                         const registry_model::CycleSlot &checked) noexcept {
  if (checked.token == 0u) {
    return checked.count == 0u && !checked.closed &&
           checked.tokens == std::array<std::uint64_t, 2u>{} &&
           checked.ordinals == std::array<std::uint64_t, 2u>{} &&
           checked.banks == std::array<std::uint32_t, 2u>{} &&
           checked.terminals ==
               std::array<std::uint64_t, 2u>{NeverUse, NeverUse} &&
           std::all_of(authority.cycle_state_.epochs.begin(),
                       authority.cycle_state_.epochs.end(),
                       [](const Authority::LeaseSlot &slot) {
                         return slot.cycle == 0u;
                       });
  }
  if (checked.count == 0u || checked.count > checked.tokens.size()) {
    return false;
  }
  for (std::size_t index = 0u; index < checked.count; ++index) {
    if (checked.tokens[index] == 0u ||
        checked.banks[index] >= cycle::BankCapacity ||
        checked.banks[index] != checked.ordinals[index] % cycle::BankCapacity ||
        std::any_of(checked.tokens.begin(), checked.tokens.begin() + index,
                    [&](const std::uint64_t token) {
                      return token == checked.tokens[index];
                    })) {
      return false;
    }
    std::size_t matches = 0u;
    for (const Authority::LeaseSlot &slot : authority.cycle_state_.epochs) {
      if (active(slot) && slot.cycle == checked.token &&
          slot.token == checked.tokens[index]) {
        ++matches;
      }
    }
    if (matches != 1u) {
      return false;
    }
  }
  for (const Authority::LeaseSlot &slot : authority.cycle_state_.epochs) {
    if (!active(slot) && slot.cycle != 0u) {
      return false;
    }
    const bool member = active(slot) && slot.cycle == checked.token;
    const bool declared = std::any_of(
        checked.tokens.begin(), checked.tokens.begin() + checked.count,
        [&](const std::uint64_t token) {
          return token != 0u && token == slot.token;
        });
    if (member != declared) {
      return false;
    }
  }
  for (std::size_t index = checked.count; index < checked.tokens.size();
       ++index) {
    if (checked.tokens[index] != 0u || checked.ordinals[index] != 0u ||
        checked.banks[index] != 0u || checked.terminals[index] != NeverUse) {
      return false;
    }
  }
  return true;
}

bool ReleaseCheck::non_epoch_cycle_clear(const Authority &authority) noexcept {
  return authority.cycle_state_.writeback.cycle == 0u &&
         std::all_of(
             authority.cycle_state_.graph_persists.begin(),
             authority.cycle_state_.graph_persists.end(),
             [](const Authority::LeaseSlot &slot) { return slot.cycle == 0u; });
}

bool ReleaseCheck::allowed(
    const Authority &authority,
    const std::span<const FrameRegion> regions) noexcept {
  if (authority.view_commit_quarantined_locked() || regions.empty()) {
    return false;
  }
  for (std::size_t left = 0u; left < regions.size(); ++left) {
    const FrameRegion region = regions[left];
    if (region.count == 0u || region.first > authority.frames_.size() ||
        region.count > authority.frames_.size() - region.first) {
      return false;
    }
    for (std::size_t index = region.first;
         index < static_cast<std::size_t>(region.first) + region.count;
         ++index) {
      const Authority::Frame &frame = authority.frames_[index];
      if (!frame.assigned || frame.tier != region.tier ||
          frame.role != region.role || frame.alias_claims != 0u) {
        return false;
      }
      if (frame.extent != 0u &&
          std::any_of(authority.frames_.begin(), authority.frames_.end(),
                      [&frame](const Authority::Frame &other) {
                        return other.assigned && other.extent == frame.extent &&
                               other.alias_claims != 0u;
                      })) {
        return false;
      }
    }
    for (std::size_t right = left + 1u; right < regions.size(); ++right) {
      const FrameRegion other = regions[right];
      if (other.count == 0u || other.first > authority.frames_.size() ||
          other.count > authority.frames_.size() - other.first ||
          !(static_cast<std::size_t>(region.first) + region.count <=
                other.first ||
            static_cast<std::size_t>(other.first) + other.count <=
                region.first)) {
        return false;
      }
    }
  }
  if (!execution(authority, regions, authority.execution_state_.slot) ||
      !slot(authority, regions, authority.cycle_state_.writeback) ||
      !graph_rows(authority) ||
      !std::all_of(authority.cycle_state_.epochs.begin(),
                   authority.cycle_state_.epochs.end(),
                   [&](const Authority::LeaseSlot &checked) {
                     return slot(authority, regions, checked);
                   }) ||
      !non_epoch_cycle_clear(authority) ||
      !cycle(authority, regions, authority.cycle_state_.cycle)) {
    return false;
  }
  return true;
}

} // namespace rund::compute::detail::residency::release_detail
