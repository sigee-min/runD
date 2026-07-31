#pragma once

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] bool
CreateVulkanReduceScratchBuffers(VulkanAdapter &adapter,
                                 VulkanReduceEncodeResources &resources,
                                 const rund::kernel::ReducePlan &plan) {
  const rund::kernel::u64 partial_bytes = plan.partial_bytes == 0u
                                              ? plan.partial_element_bytes
                                              : plan.partial_bytes;
  return CreateVulkanBuffer(
             adapter, partial_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
             resources.partial, nullptr, VulkanMemoryUse::Scratch) &&
         CreateVulkanStatus(adapter, plan.status_bytes, resources.status);
}

} // namespace
#endif

} // namespace rund::node::accel::detail
