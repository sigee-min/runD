#pragma once

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

inline constexpr VkMemoryPropertyFlags VulkanHostCoherentFlags =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

[[nodiscard]] constexpr VkMemoryPropertyFlags
VulkanMemoryRequiredFlags(const VulkanMemoryUse use) noexcept {
  return use == VulkanMemoryUse::Staging ||
                 use == VulkanMemoryUse::ResidentHost
             ? VulkanHostCoherentFlags
             : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
}

[[nodiscard]] constexpr bool
VulkanMemoryNeedsMap(const VulkanMemoryUse use) noexcept {
  return (VulkanMemoryRequiredFlags(use) & VulkanHostCoherentFlags) ==
         VulkanHostCoherentFlags;
}

[[nodiscard]] inline bool VulkanBufferMemoryReady(
    const VulkanBuffer &buffer, const VulkanMemoryUse use) noexcept {
  const VkMemoryPropertyFlags required = VulkanMemoryRequiredFlags(use);
  return buffer.buffer != VK_NULL_HANDLE &&
         buffer.memory != VK_NULL_HANDLE && buffer.memory_use == use &&
         (buffer.memory_flags & required) == required &&
         (VulkanMemoryNeedsMap(use) ? buffer.mapped != nullptr
                                    : buffer.mapped == nullptr);
}

[[nodiscard]] inline bool
VulkanHostCoherentBufferReady(const VulkanBuffer &buffer) noexcept {
  return buffer.buffer != VK_NULL_HANDLE &&
         buffer.memory != VK_NULL_HANDLE && buffer.mapped != nullptr &&
         (buffer.memory_flags & VulkanHostCoherentFlags) ==
             VulkanHostCoherentFlags;
}

[[nodiscard]] inline bool FindVulkanMemoryType(
    const VulkanAdapter &adapter, const std::uint32_t type_bits,
    const VulkanMemoryUse use, std::uint32_t &type_index,
    VkMemoryPropertyFlags &memory_flags) {
  const VkMemoryPropertyFlags required = VulkanMemoryRequiredFlags(use);
  const VkMemoryPropertyFlags preferred =
      use == VulkanMemoryUse::Staging ? VK_MEMORY_PROPERTY_HOST_CACHED_BIT
                                      : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  const VkPhysicalDeviceMemoryProperties &props = adapter.memory_properties;
  std::uint32_t first = props.memoryTypeCount;
  for (std::uint32_t index = 0u; index < props.memoryTypeCount; ++index) {
    const bool allowed = (type_bits & (1u << index)) != 0u;
    const VkMemoryPropertyFlags flags = props.memoryTypes[index].propertyFlags;
    if (!allowed || (flags & required) != required) {
      continue;
    }
    if ((flags & preferred) == preferred) {
      type_index = index;
      memory_flags = flags;
      return true;
    }
    if (first == props.memoryTypeCount) {
      first = index;
    }
  }
  if (first == props.memoryTypeCount) {
    return false;
  }
  type_index = first;
  memory_flags = props.memoryTypes[first].propertyFlags;
  return true;
}

#endif

} // namespace rund::node::accel::detail
