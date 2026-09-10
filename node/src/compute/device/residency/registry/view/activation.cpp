#include "../freeze/internal.hpp"
#include "../view_owner.hpp"

#include "../frame.hpp"
#include "../lease_state.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>

namespace rund::compute::detail::residency {

namespace view_plan_detail {

ViewActivationResult
validate_regions(const std::vector<Authority::Frame> &frames,
                 const std::span<const FrameRegion> regions) noexcept {
  if (regions.empty()) {
    return ViewActivationResult{.failure = AuthorityFailure::Invalid};
  }
  for (std::size_t region_index = 0u; region_index < regions.size();
       ++region_index) {
    const FrameRegion region = regions[region_index];
    if (region.count == 0u || region.first > frames.size() ||
        region.count > frames.size() - region.first) {
      return ViewActivationResult{.failure = AuthorityFailure::Invalid};
    }
    const Authority::Frame &selected = frames[region.first];
    if (selected.direct_registration != nullptr) {
      return ViewActivationResult{.failure = AuthorityFailure::Busy};
    }
    if (!selected.assigned || selected.tier != region.tier ||
        selected.role != region.role || selected.view_commit_stamp != 0u) {
      return ViewActivationResult{.failure = AuthorityFailure::Invalid};
    }
    for (std::size_t index = region.first;
         index < static_cast<std::size_t>(region.first) + region.count;
         ++index) {
      const Authority::Frame &frame = frames[index];
      if (frame.direct_registration != nullptr) {
        return ViewActivationResult{.failure = AuthorityFailure::Busy};
      }
      if (!frame.assigned || frame.tier != region.tier ||
          frame.role != region.role || frame.extent != selected.extent ||
          frame.view != selected.view || frame.view_commit_stamp != 0u) {
        return ViewActivationResult{.failure = AuthorityFailure::Invalid};
      }
    }
    if (selected.extent == 0u) {
      continue;
    }
    for (std::size_t prior = 0u; prior < region_index; ++prior) {
      const Authority::Frame &other = frames[regions[prior].first];
      if (other.extent == selected.extent && other.view != selected.view) {
        return ViewActivationResult{.failure = AuthorityFailure::Invalid};
      }
    }
    for (const Authority::Frame &frame : frames) {
      if (!frame.assigned || frame.extent != selected.extent ||
          frame.view == selected.view || frame.state == FrameState::Empty) {
        continue;
      }
      if (frame.view_commit_stamp != 0u) {
        return ViewActivationResult{.failure = AuthorityFailure::Busy};
      }
      if (frame.direct_registration != nullptr) {
        continue;
      }
      if (frame.alias_claims != 0u) {
        return ViewActivationResult{.failure = AuthorityFailure::Busy};
      }
      if (!frame.dirty.empty()) {
        return ViewActivationResult{.failure = AuthorityFailure::Capacity};
      }
      if (frame.state != FrameState::Resident) {
        return ViewActivationResult{.failure = AuthorityFailure::Busy};
      }
    }
  }
  return ViewActivationResult{.failure = AuthorityFailure::None};
}

std::uint64_t
apply_regions(std::vector<Authority::Frame> &frames,
              const std::span<const FrameRegion> regions) noexcept {
  std::uint64_t evictions = 0u;
  for (const FrameRegion region : regions) {
    const Authority::Frame &selected = frames[region.first];
    if (selected.extent == 0u) {
      continue;
    }
    for (Authority::Frame &frame : frames) {
      if (frame.assigned && frame.extent == selected.extent &&
          frame.view != selected.view && frame.state != FrameState::Empty &&
          frame.alias_claims == 0u && frame.direct_registration == nullptr) {
        frame = frame_detail::empty(frame);
        ++evictions;
      }
    }
  }
  return evictions;
}

} // namespace view_plan_detail

ViewActivationResult
ViewOwner::activate_views(const std::span<const FrameRegion> regions) noexcept {
  std::lock_guard lock{authority_.gate_};
  const bool epoch_active =
      std::any_of(authority_.cycle_state_.epochs.begin(),
                  authority_.cycle_state_.epochs.end(), active);
  if (authority_.view_commit_state_locked() !=
          registry_model::ViewCommitState::Idle ||
      regions.empty() || authority_.execution_state_.slot.token != 0u ||
      active(authority_.cycle_state_.writeback) ||
      any_active(authority_.cycle_state_.graph_persists) || epoch_active) {
    return ViewActivationResult{
        .failure = active(authority_.cycle_state_.writeback) ||
                           any_active(authority_.cycle_state_.graph_persists) ||
                           epoch_active ||
                           authority_.execution_state_.slot.token != 0u
                       ? AuthorityFailure::Busy
                       : AuthorityFailure::Invalid};
  }
  const ViewActivationResult checked =
      view_plan_detail::validate_regions(authority_.frames_, regions);
  if (!checked) {
    return checked;
  }
  return ViewActivationResult{.failure = AuthorityFailure::None,
                              .eviction_count = view_plan_detail::apply_regions(
                                  authority_.frames_, regions)};
}

} // namespace rund::compute::detail::residency
