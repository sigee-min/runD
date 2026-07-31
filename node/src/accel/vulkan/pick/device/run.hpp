#pragma once

#include <accel/check.hpp>

#include "adapter.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK) &&                                      \
    defined(RUND_NODE_HAVE_GLSLANG_VALIDATOR)
[[nodiscard]] inline VulkanAdapterPick
RejectVulkanAdapterPick(const char *const reason) {
  return VulkanAdapterPick{.check = rund::AccelCheck{false, reason}};
}

[[nodiscard]] VulkanAdapterPick
PickVulkanAdapterFromInstance(const VkInstance instance) {
  std::vector<VkPhysicalDevice> physical_devices{};
  if (!EnumerateVulkanPhysicalDevices(instance, physical_devices)) {
    return RejectVulkanAdapterPick("accel_vulkan_device_unavailable");
  }

  bool found_queue = false;
  bool device_failed = false;
  bool queue_failed = false;
  for (const VkPhysicalDevice physical_device : physical_devices) {
    const VulkanCreatedDevice created =
        CreateVulkanLogicalDevice(physical_device);
    found_queue = found_queue || created.found_queue;
    device_failed = device_failed || created.device_failed;
    queue_failed = queue_failed || created.queue_failed;
    if (created.device == VK_NULL_HANDLE || created.queue == VK_NULL_HANDLE) {
      continue;
    }
    auto adapter =
        VulkanAdapterFromCreatedDevice(instance, physical_device, created);
    if (adapter == nullptr) {
      return RejectVulkanAdapterPick("accel_vulkan_command_unavailable");
    }
    return VulkanAdapterPick{.check = rund::AccelCheck{true, "ok"},
                             .adapter = std::move(adapter)};
  }

  if (!found_queue || queue_failed) {
    return RejectVulkanAdapterPick("accel_vulkan_queue_unavailable");
  }
  if (device_failed) {
    return RejectVulkanAdapterPick("accel_vulkan_device_unavailable");
  }
  return RejectVulkanAdapterPick("accel_vulkan_unavailable");
}
#endif

} // namespace rund::node::accel::detail
