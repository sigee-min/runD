#include "local.hpp"

#include <algorithm>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
bool HasVulkanExtension(const std::vector<VkExtensionProperties>& props,
                        const std::string_view name) {
  return std::any_of(props.begin(), props.end(), [&](const auto& prop) {
    return std::string_view{prop.extensionName} == name;
  });
}

bool EnumerateVulkanInstanceExtensions(
    std::vector<VkExtensionProperties>& props) {
  std::uint32_t count = 0u;
  VkResult result = vkEnumerateInstanceExtensionProperties(nullptr, &count,
                                                          nullptr);
  if (result != VK_SUCCESS) {
    return false;
  }
  props.resize(count);
  if (count == 0u) {
    return true;
  }
  result =
      vkEnumerateInstanceExtensionProperties(nullptr, &count, props.data());
  if (result != VK_SUCCESS) {
    return false;
  }
  props.resize(count);
  return true;
}

bool EnumerateVulkanDeviceExtensions(
    const VkPhysicalDevice physical_device,
    std::vector<VkExtensionProperties>& props) {
  std::uint32_t count = 0u;
  VkResult result =
      vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count,
                                           nullptr);
  if (result != VK_SUCCESS) {
    return false;
  }
  props.resize(count);
  if (count == 0u) {
    return true;
  }
  result = vkEnumerateDeviceExtensionProperties(physical_device, nullptr,
                                                &count, props.data());
  if (result != VK_SUCCESS) {
    return false;
  }
  props.resize(count);
  return true;
}
#endif

}  // namespace rund::node::accel::detail
