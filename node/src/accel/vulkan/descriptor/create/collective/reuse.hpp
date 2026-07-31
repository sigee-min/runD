#pragma once

#include "admit.hpp"
#include <rund/counter.hpp>

#include <cstddef>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool ReuseCollectiveDescriptorSet(
    VulkanAdapter& adapter,
    VulkanCollectivePipeline& pipeline,
    const std::size_t slot,
    VkDescriptorSet& set) {
  if (slot >= pipeline.reusable_descriptor_count ||
      slot >= pipeline.descriptor_sets.size() ||
      pipeline.descriptor_sets[slot] == VK_NULL_HANDLE) {
    return false;
  }
  ::rund::detail::counter::Accumulate(adapter.descriptor_reuse_hit_count, 1u);
  set = pipeline.descriptor_sets[slot];
  return true;
}

#endif

}  // namespace rund::node::accel::detail
