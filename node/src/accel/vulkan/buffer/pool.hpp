#pragma once

#include <rund/counter.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
#include <vulkan/vulkan.h>
#endif

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

inline constexpr std::size_t kVulkanPoolCapacity = 32u;

[[nodiscard]] constexpr std::uint64_t
VulkanPoolLimit(const VkDeviceSize staging_bytes) noexcept {
  constexpr std::uint64_t alignment = 4u;
  return ::rund::detail::counter::SaturatingMultiply(
      std::max<std::uint64_t>(staging_bytes, alignment),
      kVulkanPoolCapacity);
}

[[nodiscard]] constexpr bool
FitsVulkanPool(const std::uint64_t current, const std::uint64_t added,
               const std::uint64_t limit) noexcept {
  return added <= limit && current <= limit - added;
}

#endif

} // namespace rund::node::accel::detail
