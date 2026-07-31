#pragma once

#include "enumerate.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK) && \
    defined(RUND_NODE_HAVE_GLSLANG_VALIDATOR)
struct VulkanCreatedDevice {
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  std::uint32_t queue_family = 0u;
  std::uint32_t timestamp_valid_bits = 0u;
  std::vector<VkExtensionProperties> extensions{};
  bool found_queue = false;
  bool device_failed = false;
  bool queue_failed = false;
};

[[nodiscard]] inline std::vector<const char*> EnabledVulkanDeviceExtensions(
    const std::vector<VkExtensionProperties>& device_extensions) {
  std::vector<const char*> enabled{};
  if (HasVulkanExtension(device_extensions, kVulkanPortabilitySubsetExtension)) {
    enabled.push_back(kVulkanPortabilitySubsetExtension);
  }
  return enabled;
}

[[nodiscard]] inline VulkanCreatedDevice CreateVulkanLogicalDevice(
    const VkPhysicalDevice physical_device) {
  VulkanCreatedDevice out{};
  if (!FindVulkanComputeQueueFamily(physical_device, out.queue_family,
                                    out.timestamp_valid_bits)) {
    return out;
  }
  out.found_queue = true;

  VkPhysicalDeviceFeatures enabled_features{};
  if (!VulkanDeviceSupportsRequiredFeatures(physical_device,
                                            enabled_features) ||
      !EnumerateVulkanDeviceExtensions(physical_device, out.extensions)) {
    out.device_failed = true;
    return out;
  }

  const std::vector<const char*> enabled_extensions =
      EnabledVulkanDeviceExtensions(out.extensions);
  const float queue_priority = 1.0F;
  VkDeviceQueueCreateInfo queue_info{};
  queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_info.queueFamilyIndex = out.queue_family;
  queue_info.queueCount = 1u;
  queue_info.pQueuePriorities = &queue_priority;

  VkDeviceCreateInfo device_info{};
  device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_info.queueCreateInfoCount = 1u;
  device_info.pQueueCreateInfos = &queue_info;
  device_info.pEnabledFeatures = &enabled_features;
  device_info.enabledExtensionCount =
      static_cast<std::uint32_t>(enabled_extensions.size());
  device_info.ppEnabledExtensionNames = enabled_extensions.data();

  if (vkCreateDevice(physical_device, &device_info, nullptr, &out.device) !=
          VK_SUCCESS ||
      out.device == VK_NULL_HANDLE) {
    out.device_failed = true;
    return out;
  }
  vkGetDeviceQueue(out.device, out.queue_family, 0u, &out.queue);
  if (out.queue == VK_NULL_HANDLE) {
    out.queue_failed = true;
    vkDestroyDevice(out.device, nullptr);
    out.device = VK_NULL_HANDLE;
  }
  return out;
}
#endif

}  // namespace rund::node::accel::detail
