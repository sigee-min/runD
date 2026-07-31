#pragma once

#include <cstdint>
#include <limits>

namespace rund::kernel::compute_lowering_detail {

inline constexpr std::uint32_t kVulkanMapWidth = 256u;
inline constexpr std::uint32_t kVulkanMapPushBytes = 2u * sizeof(std::uint32_t);

[[nodiscard]] constexpr std::uint64_t
VulkanMapTileCapacity(const std::uint32_t max_groups) noexcept {
  const std::uint64_t capacity =
      static_cast<std::uint64_t>(max_groups) * kVulkanMapWidth;
  constexpr std::uint64_t index_capacity =
      std::numeric_limits<std::uint32_t>::max();
  return capacity < index_capacity ? capacity : index_capacity;
}

[[nodiscard]] constexpr std::uint32_t
VulkanMapGroupsForTiles(const std::uint64_t tiles) noexcept {
  if (tiles == 0u || tiles > static_cast<std::uint64_t>(
                                 std::numeric_limits<std::uint32_t>::max())) {
    return 0u;
  }
  return static_cast<std::uint32_t>(tiles / kVulkanMapWidth +
                                    (tiles % kVulkanMapWidth == 0u ? 0u : 1u));
}

} // namespace rund::kernel::compute_lowering_detail
