#pragma once

#include "queue.hpp"

#include <rund/counter.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
std::uint64_t VulkanDeviceMemoryBytes(
    const VkPhysicalDevice physical_device,
    const VkPhysicalDeviceMemoryProperties &props,
    const std::vector<VkExtensionProperties> &device_extensions) {
#if defined(VK_EXT_memory_budget)
  if (physical_device != VK_NULL_HANDLE &&
      HasVulkanExtension(device_extensions,
                         VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    budget.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    VkPhysicalDeviceMemoryProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    properties.pNext = &budget;
    vkGetPhysicalDeviceMemoryProperties2(physical_device, &properties);
    std::uint64_t device_budget = 0u;
    std::uint64_t total_budget = 0u;
    for (std::uint32_t index = 0u;
         index < properties.memoryProperties.memoryHeapCount; ++index) {
      const std::uint64_t bytes = budget.heapBudget[index];
      total_budget =
          ::rund::detail::counter::SaturatingAdd(total_budget, bytes);
      if ((properties.memoryProperties.memoryHeaps[index].flags &
           VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0u) {
        device_budget =
            ::rund::detail::counter::SaturatingAdd(device_budget, bytes);
      }
    }
    if (device_budget != 0u) {
      return device_budget;
    }
    if (total_budget != 0u) {
      return total_budget;
    }
  }
#else
  static_cast<void>(physical_device);
  static_cast<void>(device_extensions);
#endif
  std::uint64_t device_local = 0u;
  std::uint64_t total = 0u;
  for (std::uint32_t index = 0u; index < props.memoryHeapCount; ++index) {
    const std::uint64_t heap_size =
        static_cast<std::uint64_t>(props.memoryHeaps[index].size);
    total = ::rund::detail::counter::SaturatingAdd(total, heap_size);
    if ((props.memoryHeaps[index].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) !=
        0u) {
      device_local =
          ::rund::detail::counter::SaturatingAdd(device_local, heap_size);
    }
  }
  if (device_local != 0u) {
    return device_local;
  }
  if (total != 0u) {
    return total;
  }
  return 64u * kVulkanOneMiB;
}
#endif

} // namespace rund::node::accel::detail
