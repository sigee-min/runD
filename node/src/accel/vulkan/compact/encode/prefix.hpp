#pragma once

#include "classify.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

void EncodeVulkanCompactPrefix(const VulkanCompactEncodeResources &compact,
                               const VkCommandBuffer command) {
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     compact.prefix_pipeline->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        compact.prefix_pipeline->pipeline_layout, 0u, 1u,
                        &compact.prefix_set, 0u, nullptr);
  DispatchVulkan(command, 1u, 1u, 1u);
}

void EncodeVulkanCompactOffsetsBarrier(
    const VulkanCompactEncodeResources &compact,
    const VkCommandBuffer command) {
  const VkBufferMemoryBarrier barrier = VulkanBufferBarrier(
      compact.offsets, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                       1u, &barrier, 0u, nullptr);
}

} // namespace
#endif

} // namespace rund::node::accel::detail
