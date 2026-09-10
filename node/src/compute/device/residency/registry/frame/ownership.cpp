#include "../../registry.hpp"

#include "../frame.hpp"
#include "../release_check.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>

namespace rund::compute::detail::residency {

bool Authority::release_frames(
    const std::span<const FrameRegion> regions) noexcept {
  std::lock_guard lock{gate_};
  if (view_commit_state_locked() != registry_model::ViewCommitState::Idle ||
      !release_detail::ReleaseCheck::allowed(*this, regions)) {
    return false;
  }

  const auto releasable = [this](const std::size_t index) {
    const Frame &frame = frames_[index];
    // Region destruction may forget only clean cached bytes. Dirty bytes are
    // the sole newer copy, and Mapping/Pinned/Writeback without an active
    // lease is an invariant failure rather than permission to erase state.
    return frame.assigned && frame.alias_claims == 0u && frame.dirty.empty() &&
           (frame.state == FrameState::Empty ||
            frame.state == FrameState::Resident);
  };
  for (const FrameRegion region : regions) {
    for (std::size_t index = region.first;
         index < static_cast<std::size_t>(region.first) + region.count;
         ++index) {
      if (!releasable(index)) {
        return false;
      }
    }
  }
  for (const FrameRegion region : regions) {
    for (std::size_t index = region.first;
         index < static_cast<std::size_t>(region.first) + region.count;
         ++index) {
      frames_[index] = Frame{};
    }
  }
  return true;
}

bool Authority::owns_regions(
    const std::span<const FrameRegion> regions) const noexcept {
  std::lock_guard lock{gate_};
  if (view_commit_state_locked() != registry_model::ViewCommitState::Idle ||
      regions.empty()) {
    return false;
  }
  for (std::size_t left = 0u; left < regions.size(); ++left) {
    const FrameRegion region = regions[left];
    const std::size_t first = region.first;
    const std::size_t count = region.count;
    if (count == 0u || first > frames_.size() ||
        count > frames_.size() - first) {
      return false;
    }
    for (std::size_t index = first; index < first + count; ++index) {
      const Frame &frame = frames_[index];
      if (!frame.assigned || frame.tier != region.tier ||
          frame.role != region.role) {
        return false;
      }
    }
    for (std::size_t right = left + 1u; right < regions.size(); ++right) {
      const FrameRegion other = regions[right];
      const std::size_t other_first = other.first;
      const std::size_t other_count = other.count;
      if (other_count == 0u || other_first > frames_.size() ||
          other_count > frames_.size() - other_first ||
          !(first + count <= other_first ||
            other_first + other_count <= first)) {
        return false;
      }
    }
  }
  return true;
}

} // namespace rund::compute::detail::residency
