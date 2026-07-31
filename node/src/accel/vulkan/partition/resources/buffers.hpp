#pragma once

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] bool
CreateVulkanPartitionScratchBuffers(VulkanAdapter &adapter,
                                    VulkanPartitionEncodeResources &resources) {
  const VkDeviceSize element_bytes = static_cast<VkDeviceSize>(
      resources.plan.element_count * sizeof(rund::kernel::u32));
  const VkDeviceSize totals_bytes = static_cast<VkDeviceSize>(
      resources.scan_plan.block_count * sizeof(rund::kernel::u32));
  return CreateVulkanBuffer(adapter, sizeof(PartitionParams),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            resources.params) &&
         CreateVulkanBuffer(
             adapter, element_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
             resources.false_bits, nullptr, VulkanMemoryUse::Scratch) &&
         CreateVulkanBuffer(
             adapter, element_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
             resources.false_offsets, nullptr, VulkanMemoryUse::Scratch) &&
         CreateVulkanBuffer(
             adapter, totals_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
             resources.false_totals, nullptr, VulkanMemoryUse::Scratch) &&
         CreateVulkanStatus(adapter, sizeof(rund::kernel::u32),
                            resources.false_status);
}

} // namespace
#endif

} // namespace rund::node::accel::detail
