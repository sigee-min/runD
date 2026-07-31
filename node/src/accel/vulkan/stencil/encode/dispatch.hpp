#pragma once

#include "state.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

void EncodeVulkanStencilDispatch(const VulkanStencilEncodeResources &stencil,
                                 const VkCommandBuffer command,
                                 const std::uint32_t workgroups) {
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     stencil.pipeline->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        stencil.pipeline->pipeline_layout, 0u, 1u,
                        &stencil.descriptor_set, 0u, nullptr);
  DispatchVulkan(command, workgroups, 1u, 1u);
}

} // namespace
#endif

} // namespace rund::node::accel::detail
