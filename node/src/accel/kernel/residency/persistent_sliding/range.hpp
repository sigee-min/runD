#pragma once

#include "capability.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace rund::node::accel::detail {

// Validate the complete coordinate range once for both request and native
// preparation paths. The last turn is checked independently for every active
// slot so control and descriptor generations cannot wrap; UINT64_MAX remains
// the no-coordinate sentinel, not a descriptor-generation limit.
template <typename Role>
[[nodiscard]] inline bool persistent_sliding_range_valid(
    const std::span<const Role> roles, const std::size_t width,
    const std::uint64_t coordinates) noexcept {
  constexpr std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
  if (coordinates == 0u || coordinates == max || width == 0u ||
      width > PersistentResidencySlidingCapacity || width > roles.size()) {
    return false;
  }
  for (std::size_t slot = 0u; slot < width; ++slot) {
    if (static_cast<std::uint64_t>(slot) >= coordinates) {
      continue;
    }
    const Role &role = roles[slot];
    if (role.first_control_generation == 0u ||
        role.control_generation_stride == 0u ||
        role.first_descriptor_generation == 0u ||
        role.descriptor_generation_stride == 0u) {
      return false;
    }
    const std::uint64_t turn =
        (coordinates - 1u - static_cast<std::uint64_t>(slot)) / width;
    if (turn >
            (std::numeric_limits<std::uint32_t>::max() -
             role.first_control_generation) /
                role.control_generation_stride ||
        turn >
            (max - role.first_descriptor_generation) /
                role.descriptor_generation_stride) {
      return false;
    }
  }
  return true;
}

} // namespace rund::node::accel::detail
