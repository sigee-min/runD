#pragma once

#include "admit.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool ReserveCollectiveDescriptorStorage(
    VulkanAdapter& adapter,
    VulkanCollectivePipeline& pipeline,
    const std::size_t target) {
  try {
    pipeline.descriptor_sets.reserve(target);
    pipeline.descriptor_leased.reserve(target);
    pipeline.descriptor_pools.reserve(
        pipeline.descriptor_pools.size() + 1u);
    return true;
  } catch (...) {
    SetVulkanLastError(adapter, "accel_vulkan_descriptor_unavailable");
    return false;
  }
}

#endif

}  // namespace rund::node::accel::detail
