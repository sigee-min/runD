#pragma once

#include "state.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

void EncodeVulkanSortDispatch(const VulkanSortEncodeResources &sort,
                              const VkCommandBuffer command) {
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     sort.dispatch_pipeline->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        sort.dispatch_pipeline->pipeline_layout, 0u, 1u,
                        &sort.dispatch_descriptor, 0u, nullptr);
  DispatchVulkan(command, 1u, 1u, 1u);
  const VkBufferMemoryBarrier barrier =
      VulkanBufferBarrier(sort.dispatch_args, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0u, 0u, nullptr, 1u,
                       &barrier, 0u, nullptr);
}

} // namespace
#endif

} // namespace rund::node::accel::detail
