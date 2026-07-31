#pragma once

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
bool FindVulkanComputeQueueFamily(
    const VkPhysicalDevice physical_device,
    std::uint32_t& queue_family,
    std::uint32_t& timestamp_valid_bits) {
  std::uint32_t count = 0u;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, nullptr);
  if (count == 0u) { return false; }
  std::vector<VkQueueFamilyProperties> props(count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count,
                                           props.data());
  props.resize(count);
  for (std::uint32_t index = 0u; index < count; ++index) {
    if ((props[index].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0u &&
        props[index].queueCount > 0u) {
      queue_family = index;
      timestamp_valid_bits = props[index].timestampValidBits;
      return true;
    }
  }
  return false;
}
#endif

}  // namespace rund::node::accel::detail
