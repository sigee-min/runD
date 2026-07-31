#pragma once

#include "../../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK) &&                                      \
    defined(RUND_NODE_HAVE_GLSLANG_VALIDATOR)
[[nodiscard]] inline bool EnumerateVulkanPhysicalDevices(
    const VkInstance instance,
    std::vector<VkPhysicalDevice> &physical_devices) {
  std::uint32_t physical_count = 0u;
  VkResult result =
      vkEnumeratePhysicalDevices(instance, &physical_count, nullptr);
  if (result != VK_SUCCESS || physical_count == 0u) {
    return false;
  }
  physical_devices.resize(physical_count);
  result = vkEnumeratePhysicalDevices(instance, &physical_count,
                                      physical_devices.data());
  if (result != VK_SUCCESS || physical_count == 0u) {
    return false;
  }
  physical_devices.resize(physical_count);
  return true;
}
#endif

} // namespace rund::node::accel::detail
