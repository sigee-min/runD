#pragma once

#include "prefix.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

void EncodeVulkanCompactScatter(const VulkanCompactEncodeResources &compact,
                                const VkCommandBuffer command,
                                const std::uint32_t workgroups) {
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     compact.scatter_pipeline->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        compact.scatter_pipeline->pipeline_layout, 0u, 1u,
                        &compact.scatter_set, 0u, nullptr);
  DispatchVulkan(command, workgroups, 1u, 1u);
}

} // namespace
#endif

} // namespace rund::node::accel::detail
