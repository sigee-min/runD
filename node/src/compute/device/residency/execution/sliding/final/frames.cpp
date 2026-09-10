#include "internal.hpp"

#include "../../../registry/frame.hpp"

namespace rund::compute::detail::residency {
void SlidingOwner::clear_sliding_frame_observation(
    const std::uint32_t frame) noexcept {
  for (std::size_t index = 0u;
       index < authority_.execution_state_.slot.frame_count; ++index) {
    if (authority_.execution_state_.slot.frames[index] == frame) {
      authority_.execution_state_.slot.observed_frame_keys[index] = {};
      authority_.execution_state_.slot.observed_frame_valid[index] = false;
      return;
    }
  }
}

void SlidingOwner::observe_sliding_frame(const std::uint32_t frame,
                                         const CacheKey key) noexcept {
  for (std::size_t index = 0u;
       index < authority_.execution_state_.slot.frame_count; ++index) {
    if (authority_.execution_state_.slot.frames[index] == frame) {
      authority_.execution_state_.slot.observed_frame_keys[index] = key;
      authority_.execution_state_.slot.observed_frame_valid[index] =
          key.domain == CacheDomain::Backing && key.backing != 0u;
      return;
    }
  }
}

bool SlidingOwner::commit_sliding_frames(const execution::Plan *const plan,
                                         const bool success) noexcept {
  if (authority_.view_commit_quarantined_locked()) {
    return false;
  }
  // Known failure keeps the established full-clear disposition. Unknown
  // terminals never reach this helper: their prepared Final is rejected
  // before Authority mutation and the existing quarantine owner remains
  // authoritative.
  for (std::size_t index = 0u;
       index < authority_.execution_state_.slot.frame_count; ++index) {
    if (authority_.execution_state_.slot.frames[index] >=
        authority_.frames_.size()) {
      return false;
    }
  }
  if (!success) {
    for (std::size_t index = 0u;
         index < authority_.execution_state_.slot.frame_count; ++index) {
      const std::uint32_t frame =
          authority_.execution_state_.slot.frames[index];
      authority_.frames_[frame] =
          frame_detail::empty(authority_.frames_[frame]);
    }
    return true;
  }

  std::array<bool, registry_model::ExecutionFrameCapacity> retained{};
  for (std::size_t index = 0u;
       index < authority_.execution_state_.slot.frame_count; ++index) {
    const std::uint32_t frame = authority_.execution_state_.slot.frames[index];
    if (frame >= authority_.frames_.size() ||
        std::find(authority_.execution_state_.slot.frames.begin(),
                  authority_.execution_state_.slot.frames.begin() +
                      static_cast<std::ptrdiff_t>(index),
                  frame) != authority_.execution_state_.slot.frames.begin() +
                                static_cast<std::ptrdiff_t>(index)) {
      return false;
    }
  }
  const auto execution_slot = [this](const std::size_t frame) noexcept {
    return std::find(authority_.execution_state_.slot.frames.begin(),
                     authority_.execution_state_.slot.frames.begin() +
                         static_cast<std::ptrdiff_t>(
                             authority_.execution_state_.slot.frame_count),
                     static_cast<std::uint32_t>(frame)) -
           authority_.execution_state_.slot.frames.begin();
  };

  const auto exact_plan_key = [plan](
                                  const registry_model::Frame &frame) noexcept {
    if (frame.key.domain != CacheDomain::Backing || frame.key.backing == 0u) {
      return false;
    }
    if (plan == nullptr) {
      // Raw Sliding close has already authenticated each fetch source against
      // its sealed Plan before the frame became Resident. Keep this final
      // check allocation-free while still rejecting an empty/stale key.
      return true;
    }
    execution::FetchSource source{};
    const bool projected =
        plan->has_window_footprint()
            ? plan->canonical_input_source(frame.key.page, source)
            : plan->input_source(frame.key.page, source);
    return projected && source.key == frame.key && source.materializes_frame();
  };

  const auto &host_regions =
      authority_.execution_state_.slot.sliding_host_input_regions;
  const std::size_t active_banks =
      static_cast<std::size_t>(std::min<std::uint64_t>(
          authority_.execution_state_.slot.epochs, execution::BankCapacity));
  for (std::size_t bank = 0u; bank < active_banks; ++bank) {
    const FrameRegion region = host_regions[bank];
    if (region.count == 0u) {
      continue;
    }
    if (region.tier != FrameTier::Host || region.role != FrameRole::Input ||
        region.first > authority_.frames_.size() ||
        region.count > authority_.frames_.size() - region.first) {
      return false;
    }
    for (std::size_t local = 0u; local < region.count; ++local) {
      const std::size_t frame_index =
          static_cast<std::size_t>(region.first) + local;
      const std::size_t slot = execution_slot(frame_index);
      if (slot >= authority_.execution_state_.slot.frame_count ||
          retained[slot]) {
        return false;
      }
      registry_model::Frame &frame = authority_.frames_[frame_index];
      if (frame.tier != FrameTier::Host || frame.role != FrameRole::Input ||
          !frame.assigned || frame.alias_claims != 0u) {
        return false;
      }
      if (frame.state == FrameState::Empty) {
        if (frame.key != CacheKey{} || !frame.dirty.empty()) {
          return false;
        }
        continue;
      }
      if (frame.state != FrameState::Resident || !frame.dirty.empty()) {
        return false;
      }
      // A clean row with a different sealed Plan key is still safe cache
      // state, but it is not part of this run's exact projection and is
      // retired below rather than being carried into the next generation.
      retained[slot] =
          plan == nullptr
              ? authority_.execution_state_.slot.observed_frame_valid[slot] &&
                    authority_.execution_state_.slot
                            .observed_frame_keys[slot] == frame.key
              : exact_plan_key(frame);
    }
  }

  // No callback, transfer, writeback, or alias may remain live at a
  // successful Sliding Final. Validate every reserved row before changing
  // any one of them so a malformed receipt cannot leave a partial cache
  // commit.
  for (std::size_t index = 0u;
       index < authority_.execution_state_.slot.frame_count; ++index) {
    const std::uint32_t frame_index =
        authority_.execution_state_.slot.frames[index];
    const registry_model::Frame &frame = authority_.frames_[frame_index];
    if (frame.alias_claims != 0u || !frame.dirty.empty() ||
        frame.state == FrameState::Mapping ||
        frame.state == FrameState::Pinned ||
        frame.state == FrameState::Writeback ||
        frame.state == FrameState::Dirty) {
      return false;
    }
    if (retained[index] &&
        (frame.state != FrameState::Resident || !exact_plan_key(frame))) {
      return false;
    }
  }

  for (std::size_t index = 0u;
       index < authority_.execution_state_.slot.frame_count; ++index) {
    const std::uint32_t frame_index =
        authority_.execution_state_.slot.frames[index];
    if (retained[index]) {
      registry_model::Frame &frame = authority_.frames_[frame_index];
      frame.state = FrameState::Resident;
      frame.next_use = NeverUse;
      frame.retain_until = NeverUse;
    } else {
      authority_.frames_[frame_index] =
          frame_detail::empty(authority_.frames_[frame_index]);
    }
  }
  return true;
}

} // namespace rund::compute::detail::residency
