#pragma once

#include <accel/check.hpp>

#include "classify.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::AccelCheck
EncodeVulkanPartitionScans(VulkanAdapter &adapter,
                           const VulkanPartitionEncodeResources &partition,
                           void *const command_raw) {
  return EncodeVulkanScanBuffers(adapter, partition.false_scan_resources,
                                 command_raw);
}

void EncodeVulkanPartitionScatter(
    const VulkanPartitionEncodeResources &partition,
    const VkCommandBuffer command, const std::uint32_t workgroups) {
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     partition.scatter_pipeline->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        partition.scatter_pipeline->pipeline_layout, 0u, 1u,
                        &partition.scatter_set, 0u, nullptr);
  DispatchVulkan(command, workgroups, 1u, 1u);
}

void EncodeVulkanPartitionOutputBarrier(
    const VulkanPartitionEncodeResources &partition,
    const VkCommandBuffer command) {
  const VkBufferMemoryBarrier barrier =
      VulkanDeviceOutputBarrier(*partition.output);
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       kVulkanDeviceOutputStage, 0u, 0u, nullptr, 1u, &barrier,
                       0u, nullptr);
}

} // namespace
#endif

} // namespace rund::node::accel::detail
