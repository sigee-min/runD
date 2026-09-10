#include "../release_check.hpp"

#include <algorithm>

namespace rund::compute::detail::residency::release_detail {

bool ReleaseCheck::execution(
    const Authority &authority, const std::span<const FrameRegion> regions,
    const registry_model::ExecutionSlot &checked) noexcept {
  if (checked.token == 0u) {
    return checked.direct_proof_owner == nullptr &&
           checked.registration_state == nullptr &&
           checked.registration_nonce == 0u &&
           checked.direct_binding_count == 0u && !checked.direct_released &&
           !checked.direct_success && checked.frame_count == 0u &&
           checked.undo_count == 0u && checked.native_inflight == 0u &&
           std::all_of(checked.service_binding_count.begin(),
                       checked.service_binding_count.end(),
                       [](const std::size_t count) { return count == 0u; }) &&
           std::all_of(checked.service_transition_count.begin(),
                       checked.service_transition_count.end(),
                       [](const std::size_t count) { return count == 0u; }) &&
           std::all_of(checked.window_service_binding_count.begin(),
                       checked.window_service_binding_count.end(),
                       [](const auto &bank) {
                         return std::all_of(bank.begin(), bank.end(),
                                            [](const std::size_t count) {
                                              return count == 0u;
                                            });
                       }) &&
           std::all_of(checked.window_service_transition_count.begin(),
                       checked.window_service_transition_count.end(),
                       [](const auto &bank) {
                         return std::all_of(bank.begin(), bank.end(),
                                            [](const std::size_t count) {
                                              return count == 0u;
                                            });
                       }) &&
           std::all_of(
               checked.sliding_host_input_regions.begin(),
               checked.sliding_host_input_regions.end(),
               [](const FrameRegion region) { return region.count == 0u; }) &&
           !checked.failed && !checked.unknown && !checked.native_accepted &&
           !checked.native_rejected && !checked.cache_admitted &&
           !checked.window_cache_admitted && !checked.sliding_admitted &&
           !checked.direct_recurrence_admitted && !checked.output_admitted &&
           !checked.window_final;
  }
  if (checked.plan == 0u || checked.generation == 0u || checked.epochs == 0u ||
      checked.direct_recurrence_admitted !=
          (checked.direct_proof_owner != nullptr) ||
      (checked.direct_recurrence_admitted &&
       (checked.registration_state == nullptr ||
        checked.registration_nonce == 0u ||
        checked.registration_nonce != checked.registration_state->nonce() ||
        checked.direct_binding_count == 0u ||
        checked.direct_binding_count > checked.direct_bindings.size())) ||
      checked.frame_count == 0u ||
      checked.frame_count > checked.frames.size() ||
      checked.undo_count > checked.undo_frames.size()) {
    return false;
  }
  bool has_rows = false;
  for (std::size_t index = 0u; index < checked.frame_count; ++index) {
    if (!row(authority, regions, checked.frames[index], has_rows)) {
      return false;
    }
  }
  for (std::size_t index = 0u; index < checked.undo_count; ++index) {
    if (!row(authority, regions, checked.undo_frames[index], has_rows)) {
      return false;
    }
  }
  if (checked.direct_recurrence_admitted) {
    const registration_detail::Lifecycle phase =
        checked.registration_state->phase();
    if (phase == registration_detail::Lifecycle::Active ||
        phase == registration_detail::Lifecycle::Released ||
        phase == registration_detail::Lifecycle::Quarantined) {
      return false;
    }
    for (std::size_t index = 0u; index < checked.direct_binding_count;
         ++index) {
      if (!direct_binding(authority, checked.direct_bindings[index]) ||
          authority.frames_[checked.direct_bindings[index].region.first]
                  .direct_registration != checked.registration_state ||
          !row(authority, regions,
               checked.direct_bindings[index].region.first, has_rows)) {
        return false;
      }
    }
  }
  for (std::size_t bank = 0u; bank < checked.service_bindings.size(); ++bank) {
    if (checked.service_binding_count[bank] >
            checked.service_bindings[bank].size() ||
        checked.service_transition_count[bank] >
            checked.service_transitions[bank].size()) {
      return false;
    }
    for (std::size_t index = 0u; index < checked.service_binding_count[bank];
         ++index) {
      if (!row(authority, regions,
               checked.service_bindings[bank][index].frame, has_rows)) {
        return false;
      }
    }
    for (std::size_t index = 0u;
         index < checked.service_transition_count[bank]; ++index) {
      if (!row(authority, regions,
               checked.service_transitions[bank][index].frame, has_rows)) {
        return false;
      }
    }
  }
  for (std::size_t epoch = 0u; epoch < checked.window_service_bindings.size();
       ++epoch) {
    for (std::size_t bank = 0u;
         bank < checked.window_service_bindings[epoch].size(); ++bank) {
      if (checked.window_service_binding_count[epoch][bank] >
              checked.window_service_bindings[epoch][bank].size() ||
          checked.window_service_transition_count[epoch][bank] >
              checked.window_service_transitions[epoch][bank].size()) {
        return false;
      }
      for (std::size_t index = 0u;
           index < checked.window_service_binding_count[epoch][bank]; ++index) {
        if (!row(authority, regions,
                 checked.window_service_bindings[epoch][bank][index].frame,
                 has_rows)) {
          return false;
        }
      }
      for (std::size_t index = 0u;
           index < checked.window_service_transition_count[epoch][bank];
           ++index) {
        if (!row(authority, regions,
                 checked.window_service_transitions[epoch][bank][index].frame,
                 has_rows)) {
          return false;
        }
      }
    }
  }
  bool has_sliding_region = false;
  for (const FrameRegion region : checked.sliding_host_input_regions) {
    if (region.count == 0u) {
      continue;
    }
    has_sliding_region = true;
    if (!ReleaseCheck::region(authority, regions, region, has_rows)) {
      return false;
    }
  }
  return has_rows && (!checked.sliding_admitted || has_sliding_region);
}

} // namespace rund::compute::detail::residency::release_detail
