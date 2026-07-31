#pragma once

#include "state.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

void EncodeVulkanGatherDispatch(const VulkanGatherEncodeResources &gather,
                                const VkCommandBuffer command) {
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     gather.control_pipeline->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        gather.control_pipeline->pipeline_layout, 0u, 1u,
                        &gather.control_descriptor, 0u, nullptr);
  DispatchVulkan(command, 1u, 1u, 1u);
  const std::array<VkBufferMemoryBarrier, 2u> barriers{
      VulkanBufferBarrier(gather.indirect, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_INDIRECT_COMMAND_READ_BIT |
                              VK_ACCESS_SHADER_READ_BIT),
      VulkanBufferBarrier(gather.status.device, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT)};
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                       0u, 0u, nullptr, barriers.size(), barriers.data(), 0u,
                       nullptr);
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     gather.gather_pipeline->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        gather.gather_pipeline->pipeline_layout, 0u, 1u,
                        &gather.gather_descriptor, 0u, nullptr);
  DispatchVulkanIndirect(command, gather.indirect.buffer, 0u);
}

} // namespace
#endif

} // namespace rund::node::accel::detail
