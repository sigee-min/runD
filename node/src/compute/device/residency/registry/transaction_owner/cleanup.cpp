#include "local.hpp"

#include "../frame.hpp"
#include "../lease_state.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <mutex>

namespace rund::compute::detail::residency {

bool VirtualTransactionOwner::cleanup_virtual_transaction_rows(
    const std::uint64_t backing, const std::uint64_t version,
    const std::uint64_t materialization_hi,
    const std::uint64_t materialization_lo, const std::uint64_t page_count,
    const std::uint64_t boundary_page, const std::uint64_t boundary_extent,
    const std::span<const FrameRegion> regions,
    const std::uint64_t owner_generation) const noexcept {
  if (backing == 0u || version == 0u || page_count == 0u ||
      owner_generation == 0u ||
      (boundary_page != std::numeric_limits<std::uint64_t>::max() &&
       boundary_page >= page_count) ||
      regions.size() != VirtualTransactionLease::RegionCapacity) {
    return false;
  }
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked() ||
      authority_.execution_state_.slot.token != 0u ||
      any_active(authority_.cycle_state_.epochs) ||
      active(authority_.cycle_state_.writeback) ||
      retry_ready(authority_.cycle_state_.graph_persists) ||
      !transaction_detail::valid_regions(authority_.frames_, regions)) {
    return false;
  }
  const auto exact = [&](const Authority::Frame &frame,
                         const FrameRegion region) noexcept {
    return frame.transaction_generation == owner_generation && frame.assigned &&
           frame.tier == region.tier && frame.role == FrameRole::Output &&
           frame.alias_claims == 0u && frame.state != FrameState::Empty &&
           transaction_detail::virtual_key(
               frame.key, backing, version, materialization_hi,
               materialization_lo, page_count, boundary_page, boundary_extent);
  };
  std::array<std::uint32_t, VirtualTransactionLease::Capacity> targets{};
  std::size_t target_count = 0u;
  for (const FrameRegion region : regions) {
    for (std::size_t index = region.first;
         index < static_cast<std::size_t>(region.first) + region.count;
         ++index) {
      const Authority::Frame &frame = authority_.frames_[index];
      if (frame.transaction_generation != owner_generation) {
        continue;
      }
      if (!exact(frame, region) ||
          target_count >= VirtualTransactionLease::Capacity) {
        return false;
      }
      targets[target_count++] = static_cast<std::uint32_t>(index);
    }
  }
  for (std::size_t index = 0u; index < target_count; ++index) {
    const std::uint32_t frame = targets[index];
    const auto region = std::find_if(
        regions.begin(), regions.end(), [frame](const FrameRegion candidate) {
          return frame >= candidate.first &&
                 static_cast<std::uint64_t>(frame - candidate.first) <
                     candidate.count;
        });
    if (region == regions.end() || frame >= authority_.frames_.size() ||
        !exact(authority_.frames_[frame], *region)) {
      return false;
    }
  }
  for (std::size_t index = 0u; index < target_count; ++index) {
    const std::uint32_t frame = targets[index];
    authority_.frames_[frame] = frame_detail::empty(authority_.frames_[frame]);
  }
  return true;
}

} // namespace rund::compute::detail::residency
