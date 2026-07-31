#pragma once

#include "lookup.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] bool
CreateVulkanSegmentedScanBuffers(VulkanAdapter &adapter,
                                 VulkanSegmentedScanEncodeResources &resources,
                                 const SegmentedScanParams &params_value) {
  const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  const rund::kernel::u64 value_bytes =
      resources.plan.block_count * resources.plan.element_bytes;
  const rund::kernel::u64 head_bytes =
      resources.plan.block_count * sizeof(rund::kernel::u32);
  return CreateVulkanBuffer(adapter, sizeof(params_value), storage,
                                resources.params) &&
         UploadVulkanBuffer(resources.params, &params_value,
                            sizeof(params_value)) &&
         CreateVulkanBuffer(adapter, value_bytes, storage,
                            resources.offsets, nullptr,
                            VulkanMemoryUse::Scratch) &&
         CreateVulkanBuffer(adapter, head_bytes, storage,
                            resources.first_heads, nullptr,
                            VulkanMemoryUse::Scratch) &&
         CreateVulkanStatus(adapter, head_bytes, resources.status);
}

} // namespace
#endif

} // namespace rund::node::accel::detail
