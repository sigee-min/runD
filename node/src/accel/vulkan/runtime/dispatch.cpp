#include "../map/dispatch.hpp"
#include "local.hpp"
#include "timestamp.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
bool EncodeSubmitVulkanWindow(VulkanAdapter &adapter,
                              VulkanCachedPipeline &pipeline,
                              const rund::kernel::ComputeDispatchWindow &window,
                              const VkDescriptorSet descriptor_set,
                              const VulkanWindowBuffers &buffers) {
  if (!BeginVulkanCommand(adapter)) {
    return false;
  }
  VkCommandBuffer command_buffer = adapter.command_buffer;
  BeginVulkanTimestampSpan(adapter, command_buffer);
  EncodeVulkanMap(command_buffer, pipeline, descriptor_set,
                  static_cast<std::uint32_t>(window.tile_count), 1u);
  EndVulkanTimestampSpan(adapter, command_buffer);
  if (!buffers.resident) {
    VkBufferMemoryBarrier output_barrier{};
    output_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    output_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    output_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    output_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    output_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    output_barrier.buffer = buffers.output.buffer.buffer;
    output_barrier.size = buffers.output.used_bytes;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0u, 0u, nullptr, 1u,
                         &output_barrier, 0u, nullptr);
  }
  return SubmitVulkanCommand(adapter);
}
#endif

} // namespace rund::node::accel::detail
