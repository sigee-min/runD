#pragma once

#include "properties.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
bool VulkanDeviceSupportsDriverProperties(
    const VkPhysicalDevice physical_device,
    const std::vector<VkExtensionProperties> &device_extensions) {
  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(physical_device, &props);
  const std::uint32_t major = VK_VERSION_MAJOR(props.apiVersion);
  const std::uint32_t minor = VK_VERSION_MINOR(props.apiVersion);
  return major > 1u || (major == 1u && minor >= 2u) ||
         HasVulkanExtension(device_extensions,
                            kVulkanDriverPropertiesExtension);
}

bool VulkanDeviceSupportsRequiredFeatures(
    const VkPhysicalDevice physical_device,
    VkPhysicalDeviceFeatures &features) {
  vkGetPhysicalDeviceFeatures(physical_device, &features);
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(physical_device, &properties);
  return features.shaderInt64 == VK_TRUE &&
         properties.limits.maxComputeWorkGroupInvocations >= 256u &&
         properties.limits.maxComputeWorkGroupSize[0] >= 256u &&
         properties.limits.maxComputeWorkGroupCount[0] != 0u;
}
#endif

} // namespace rund::node::accel::detail
