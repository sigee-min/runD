#pragma once

#include "../../clock.hpp"
#include "../adapter/api.hpp"
#include "../runtime/counter.hpp"

#include <cstdint>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

inline void FillVulkanDescriptorWrites(const VkDescriptorSet set,
                                       const std::uint32_t descriptor_count,
                                       VkDescriptorBufferInfo *const infos,
                                       VkWriteDescriptorSet *const writes) {
  for (std::uint32_t index = 0u; index < descriptor_count; ++index) {
    writes[index] = VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr,
        .dstSet = set,
        .dstBinding = index,
        .dstArrayElement = 0u,
        .descriptorCount = 1u,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pImageInfo = nullptr,
        .pBufferInfo = &infos[index],
        .pTexelBufferView = nullptr,
    };
  }
}

inline void
SubmitVulkanDescriptorWrites(VulkanAdapter &adapter,
                             const std::uint32_t descriptor_count,
                             const VkWriteDescriptorSet *const writes) {
  const std::uint64_t begin = MonotonicNanoseconds();
  vkUpdateDescriptorSets(adapter.device, descriptor_count, writes, 0u, nullptr);
  RecordVulkanDescriptorSetupNs(adapter, MonotonicNanoseconds() - begin);
}

#endif

} // namespace rund::node::accel::detail
