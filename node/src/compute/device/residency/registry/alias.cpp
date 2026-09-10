#include "../registry.hpp"
#include "internal.hpp"

#include <algorithm>
#include <limits>
#include <mutex>

namespace rund::compute::detail::residency {

bool Authority::issue_alias(
    const std::uint64_t owner_token,
    const std::span<const FrameRegion> source_regions,
    const FrameRegion target_region, const CacheKey source_key,
    const CacheKey target_key, const std::uint32_t target_frame,
    const std::uint64_t source_offset, const std::uint64_t target_offset,
    const std::uint64_t bytes, const std::uint64_t frame_bytes,
    AliasLease &lease) noexcept {
  lease.clear();
  if (owner_token == 0u || source_regions.empty() ||
      source_regions.size() > 2u || source_key.backing == 0u ||
      target_key.backing == 0u || bytes == 0u || frame_bytes == 0u ||
      source_offset > frame_bytes || bytes > frame_bytes - source_offset ||
      target_offset > frame_bytes || bytes > frame_bytes - target_offset) {
    return false;
  }
  std::lock_guard lock{gate_};
  if (view_commit_quarantined_locked()) {
    return false;
  }
  const auto owner =
      std::find_if(cycle_state_.epochs.begin(), cycle_state_.epochs.end(),
                   [owner_token](const LeaseSlot &slot) {
                     return slot.token == owner_token &&
                            (slot.state == LeaseState::Prepared ||
                             slot.state == LeaseState::Computing);
                   });
  if (owner == cycle_state_.epochs.end()) {
    return false;
  }
  const auto valid_region = [this](const FrameRegion region) noexcept {
    const std::size_t first = region.first;
    const std::size_t count = region.count;
    return count != 0u && first <= frames_.size() &&
           count <= frames_.size() - first && region.tier == FrameTier::Host &&
           region.role == FrameRole::Input &&
           std::all_of(frames_.begin() + static_cast<std::ptrdiff_t>(first),
                       frames_.begin() +
                           static_cast<std::ptrdiff_t>(first + count),
                       [region](const Frame &frame) {
                         return frame.assigned && frame.tier == region.tier &&
                                frame.role == region.role &&
                                frame.view_commit_stamp == 0u;
                       });
  };
  if (!valid_region(target_region) || target_frame < target_region.first ||
      target_frame - target_region.first >= target_region.count) {
    return false;
  }
  const Frame &target = frames_[target_frame];
  if (target.key != target_key || target.state != FrameState::Mapping ||
      target.alias_claims != 0u) {
    return false;
  }
  std::uint32_t source_frame = std::numeric_limits<std::uint32_t>::max();
  FrameRegion source_region{};
  for (const FrameRegion region : source_regions) {
    if (!valid_region(region)) {
      return false;
    }
    const std::size_t found =
        find_frame(frames_, source_key, region.first, region.count);
    if (found != frames_.size()) {
      if (source_frame != std::numeric_limits<std::uint32_t>::max() ||
          found > std::numeric_limits<std::uint32_t>::max()) {
        return false;
      }
      source_frame = static_cast<std::uint32_t>(found);
      source_region = region;
    }
  }
  if (source_frame == std::numeric_limits<std::uint32_t>::max() ||
      source_frame == target_frame || source_region == target_region) {
    return false;
  }
  Frame &source = frames_[source_frame];
  if (source.state != FrameState::Resident || !source.dirty.empty() ||
      source.alias_claims == std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  const std::uint64_t generation = next_token(credentials_.next_token);
  const std::uint64_t nonce = next_token(credentials_.next_token);
  if (generation == 0u || nonce == 0u) {
    return false;
  }
  ++source.alias_claims;
  lease.source_key_ = source_key;
  lease.target_key_ = target_key;
  lease.source_region_ = source_region;
  lease.target_region_ = target_region;
  lease.source_frame_ = source_frame;
  lease.target_frame_ = target_frame;
  lease.source_offset_ = source_offset;
  lease.target_offset_ = target_offset;
  lease.bytes_ = bytes;
  lease.frame_bytes_ = frame_bytes;
  lease.owner_token_ = owner_token;
  lease.generation_ = generation;
  lease.nonce_ = nonce;
  return true;
}

bool Authority::release_alias(AliasLease &&lease,
                              const bool source_known) noexcept {
  std::lock_guard lock{gate_};
  if (view_commit_quarantined_locked()) {
    return false;
  }
  const auto valid_region = [this](const FrameRegion region) noexcept {
    const std::size_t first = region.first;
    const std::size_t count = region.count;
    return count != 0u && first <= frames_.size() &&
           count <= frames_.size() - first && region.tier == FrameTier::Host &&
           region.role == FrameRole::Input &&
           std::all_of(frames_.begin() + static_cast<std::ptrdiff_t>(first),
                       frames_.begin() +
                           static_cast<std::ptrdiff_t>(first + count),
                       [region](const Frame &frame) {
                         return frame.assigned && frame.tier == region.tier &&
                                frame.role == region.role &&
                                frame.view_commit_stamp == 0u;
                       });
  };
  if (!lease || lease.owner_token_ == 0u || lease.generation_ == 0u ||
      lease.nonce_ == 0u || lease.generation_ == lease.nonce_ ||
      lease.source_key_.backing == 0u || lease.target_key_.backing == 0u ||
      lease.frame_bytes_ == 0u || lease.bytes_ == 0u ||
      lease.source_offset_ > lease.frame_bytes_ ||
      lease.bytes_ > lease.frame_bytes_ - lease.source_offset_ ||
      lease.target_offset_ > lease.frame_bytes_ ||
      lease.bytes_ > lease.frame_bytes_ - lease.target_offset_ ||
      !valid_region(lease.source_region_) ||
      !valid_region(lease.target_region_) ||
      lease.source_frame_ >= frames_.size() ||
      lease.target_frame_ >= frames_.size() ||
      lease.source_frame_ == lease.target_frame_ ||
      lease.source_frame_ < lease.source_region_.first ||
      lease.source_frame_ - lease.source_region_.first >=
          lease.source_region_.count ||
      lease.target_frame_ < lease.target_region_.first ||
      lease.target_frame_ - lease.target_region_.first >=
          lease.target_region_.count) {
    return false;
  }
  Frame &source = frames_[lease.source_frame_];
  const Frame &target = frames_[lease.target_frame_];
  if (source.key != lease.source_key_ || source.alias_claims == 0u ||
      source.state != FrameState::Resident || !source.dirty.empty() ||
      (target.state != FrameState::Empty && target.key != lease.target_key_)) {
    return false;
  }
  --source.alias_claims;
  if (!source_known) {
    source.state = FrameState::Empty;
  }
  lease.clear();
  return true;
}

} // namespace rund::compute::detail::residency
