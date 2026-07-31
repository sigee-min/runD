#pragma once

#include "state.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

void EncodeVulkanPartitionClassify(
    const VulkanPartitionEncodeResources &partition,
    const VkCommandBuffer command, const std::uint32_t workgroups) {
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     partition.classify_pipeline->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        partition.classify_pipeline->pipeline_layout, 0u, 1u,
                        &partition.classify_set, 0u, nullptr);
  DispatchVulkan(command, workgroups, 1u, 1u);
}

void EncodeVulkanPartitionClassifyBarrier(
    const VulkanPartitionEncodeResources &partition,
    const VkCommandBuffer command) {
  const VkBufferMemoryBarrier barrier =
      VulkanBufferBarrier(partition.false_bits, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT);
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                       1u, &barrier, 0u, nullptr);
}

} // namespace
#endif

} // namespace rund::node::accel::detail
