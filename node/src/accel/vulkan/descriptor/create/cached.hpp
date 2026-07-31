#pragma once

#include "layout.hpp"
#include "../../descriptor.hpp"

#include <cstdint>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

bool CreateVulkanStorageDescriptorSet(
    VulkanAdapter& adapter,
    const VulkanCachedPipeline& pipeline,
    const std::uint32_t descriptor_count,
    VkDescriptorPool& pool,
    VkDescriptorSet& set) {
  return CreateDescriptorSetWithLayout(
      adapter, descriptor_count, pipeline.descriptor_set_layout, pool, set);
}

#endif

}  // namespace rund::node::accel::detail
