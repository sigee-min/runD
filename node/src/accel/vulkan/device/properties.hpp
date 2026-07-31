#pragma once

#include "memory.hpp"

#include <algorithm>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
void PopulateVulkanDriverProperties(const VkInstance instance,
                                    const VkPhysicalDeviceProperties &props,
                                    const bool driver_properties_available,
                                    VulkanAdapter &adapter) {
  adapter.device_name = props.deviceName;
  adapter.max_dispatch_groups = props.limits.maxComputeWorkGroupCount[0];
  adapter.dispatch_rows = props.limits.maxComputeWorkGroupCount[1];
  adapter.storage_align =
      std::max<VkDeviceSize>(1u, props.limits.minStorageBufferOffsetAlignment);
  adapter.storage_limit =
      std::max<VkDeviceSize>(1u, props.limits.maxStorageBufferRange);
  adapter.timestamp_period_ns = props.limits.timestampPeriod;
  if (!driver_properties_available) {
    return;
  }

  const auto get_properties2 =
      reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
          vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2"));
  if (get_properties2 == nullptr) {
    return;
  }

  VkPhysicalDeviceDriverProperties driver_props{};
  driver_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
  VkPhysicalDeviceProperties2 props2{};
  props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  props2.pNext = &driver_props;
  get_properties2(adapter.physical_device, &props2);
  adapter.driver_name = driver_props.driverName;
  adapter.driver_info = driver_props.driverInfo;
}
#endif

} // namespace rund::node::accel::detail
