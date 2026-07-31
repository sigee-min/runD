#pragma once

#include "../../../descriptor.hpp"
#include <rund/counter.hpp>

#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool CreateDescriptorPoolForSets(
    VulkanAdapter& adapter,
    const std::uint32_t descriptor_count,
    const std::uint32_t set_count,
    VkDescriptorPool& pool) {
  if (descriptor_count == 0u || set_count == 0u ||
      descriptor_count >
          std::numeric_limits<std::uint32_t>::max() / set_count) {
    SetVulkanLastError(adapter, "accel_vulkan_descriptor_unavailable");
    return false;
  }

  VkDescriptorPoolSize pool_size{};
  pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_size.descriptorCount = descriptor_count * set_count;

  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.maxSets = set_count;
  pool_info.poolSizeCount = 1u;
  pool_info.pPoolSizes = &pool_size;
  if (vkCreateDescriptorPool(adapter.device, &pool_info, nullptr, &pool) !=
          VK_SUCCESS ||
      pool == VK_NULL_HANDLE) {
    SetVulkanLastError(adapter, "accel_vulkan_descriptor_unavailable");
    return false;
  }
  ::rund::detail::counter::Accumulate(adapter.descriptor_pool_create_count, 1u);
  return true;
}

#endif

}  // namespace rund::node::accel::detail
