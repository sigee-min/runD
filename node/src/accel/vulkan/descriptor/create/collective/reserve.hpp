#pragma once

#include "grow.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool ReserveVulkanCollectiveDescriptorSets(
    VulkanAdapter& adapter,
    VulkanCollectivePipeline& pipeline,
    const std::uint32_t descriptor_count,
    const std::uint64_t set_count) {
  if (set_count == 0u) { return true; }
  if (!CollectiveDescriptorSlotOk(adapter, pipeline, descriptor_count,
                                  set_count - 1u)) {
    return false;
  }
  return GrowCollectiveDescriptorSets(adapter, pipeline, descriptor_count,
                                      set_count);
}

#endif

}  // namespace rund::node::accel::detail
