#include "command.hpp"

#include "command/encoded.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
void EncodeVulkanComputeToComputeBarrier(const VkCommandBuffer command_buffer) {
  const VkMemoryBarrier barrier{
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
  };
  vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 1u, &barrier,
                       0u, nullptr, 0u, nullptr);
}
#endif

} // namespace rund::node::accel::detail
