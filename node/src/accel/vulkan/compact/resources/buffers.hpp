#pragma once

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] bool
CreateVulkanCompactScratchBuffers(VulkanAdapter &adapter,
                                  VulkanCompactEncodeResources &resources,
                                  const CompactParams &params_value) {
  const VkDeviceSize count_bytes =
      static_cast<VkDeviceSize>(resources.block_count) *
      sizeof(rund::kernel::u32);
  const VkDeviceSize offset_bytes =
      static_cast<VkDeviceSize>(resources.block_count) *
      sizeof(rund::kernel::u32);
  return CreateVulkanBuffer(adapter, sizeof(params_value),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                resources.params) &&
         UploadVulkanBuffer(resources.params, &params_value,
                            sizeof(params_value)) &&
         CreateVulkanBuffer(adapter, count_bytes,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            resources.counts, nullptr,
                            VulkanMemoryUse::Scratch) &&
         CreateVulkanBuffer(adapter, offset_bytes,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            resources.offsets, nullptr,
                            VulkanMemoryUse::Scratch) &&
         CreateVulkanStatus(adapter, sizeof(rund::kernel::u32),
                            resources.status);
}

} // namespace
#endif

} // namespace rund::node::accel::detail
