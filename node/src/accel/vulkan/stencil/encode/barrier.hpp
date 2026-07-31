#pragma once

#include "dispatch.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

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
