#pragma once

#include "dispatch.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

void EncodeVulkanStencilStageBarrier(
    const VulkanStencilEncodeResources &stencil, const VkCommandBuffer command,
    const std::uint32_t stage_index) {
  if (!StencilRangeUsesGlobalScratch(stencil.range)) {
    return;
  }
  const VulkanBuffer *scratch0 = nullptr;
  const VulkanBuffer *scratch1 = nullptr;
  if (!VulkanStencilStageScratchBindings(stencil, stage_index, scratch0,
                                         scratch1) ||
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

void EncodeVulkanStencilFinishBarrier(
    const VulkanStencilEncodeResources& stencil,
    const VkCommandBuffer command) {
  std::array<VkBufferMemoryBarrier, 1u> barriers{
      VulkanDeviceOutputBarrier(*stencil.output),
  };
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       kVulkanDeviceOutputStage,
                       0u, 0u, nullptr,
                       static_cast<std::uint32_t>(barriers.size()),
                       barriers.data(), 0u, nullptr);
}

}  // namespace
#endif

}  // namespace rund::node::accel::detail
