#pragma once

#include "sets.hpp"
#include "../../../descriptor.hpp"

#include <cstdint>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool CreateDescriptorSetWithLayout(
    VulkanAdapter& adapter,
    const std::uint32_t descriptor_count,
    const VkDescriptorSetLayout layout,
    VkDescriptorPool& pool,
    VkDescriptorSet& set) {
  return CreateDescriptorSetsWithLayout(adapter, descriptor_count, 1u, layout,
                                        pool, &set) &&
         set != VK_NULL_HANDLE;
}

#endif

}  // namespace rund::node::accel::detail
