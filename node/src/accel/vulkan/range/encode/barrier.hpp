#pragma once

#include "dispatch.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

void EncodeVulkanRangeControlBarrier(const VulkanRangeResources &range,
                                     const VkCommandBuffer command) {
  const std::array<VkBufferMemoryBarrier, 3u> barriers{
      VulkanBufferBarrier(range.control_params, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT),
      VulkanBufferBarrier(range.control_indirect, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_INDIRECT_COMMAND_READ_BIT),
      VulkanBufferBarrier(range.control_status.device,
                          VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT),
  };
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                       0u, 0u, nullptr,
                       static_cast<std::uint32_t>(barriers.size()),
                       barriers.data(), 0u, nullptr);
}

void EncodeVulkanRangeBarrier(const VulkanRangeResources &range,
                              const VkCommandBuffer command,
                              const std::uint32_t stage_index) {
  if (!RangeUsesScratch(range.range)) {
    return;
  }
  const VulkanBuffer *scratch0 = nullptr;
  const VulkanBuffer *scratch1 = nullptr;
  if (!VulkanRangeScratch(range, stage_index, scratch0, scratch1) ||
      scratch0 == nullptr || scratch1 == nullptr) {
    return;
  }
  std::array<VkBufferMemoryBarrier, 2u> barriers{
      VulkanBufferBarrier(*scratch0, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT |
                              VK_ACCESS_SHADER_WRITE_BIT),
      VulkanBufferBarrier(*scratch1, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT |
                              VK_ACCESS_SHADER_WRITE_BIT),
  };
  const std::uint32_t count = scratch0->buffer == scratch1->buffer &&
                                      scratch0->offset == scratch1->offset
                                  ? 1u
                                  : 2u;
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                       count, barriers.data(), 0u, nullptr);
}

[[nodiscard]] bool
EncodeVulkanRangeFinishBarrier(const VulkanRangeResources &range,
                               const VkCommandBuffer command) {
  if (range.controlled) {
    const std::array<const VulkanBuffer *, 1u> outputs{range.output};
    return FinishVulkanStatus(command, range.control_status, outputs);
  }
  std::array<VkBufferMemoryBarrier, 1u> barriers{
      VulkanDeviceOutputBarrier(*range.output),
  };
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       kVulkanDeviceOutputStage, 0u, 0u, nullptr,
                       static_cast<std::uint32_t>(barriers.size()),
                       barriers.data(), 0u, nullptr);
  return true;
}

} // namespace
#endif

} // namespace rund::node::accel::detail
