#include "../local.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <utility>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool
encode_upload(VulkanAdapter &adapter, const std::span<const UploadPlan> plans,
              ScopedBuffer &staging, const VkDeviceSize staging_bytes,
              const bool asynchronous, std::shared_ptr<void> targets) {
  if (!BeginVulkanCommand(adapter)) {
    return false;
  }
  const VkBufferMemoryBarrier source_barrier{
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .pNext = nullptr,
      .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = staging.buffer.buffer,
      .offset = 0u,
      .size = staging_bytes,
  };
  vkCmdPipelineBarrier(adapter.command_buffer, VK_PIPELINE_STAGE_HOST_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0u, 0u, nullptr, 1u,
                       &source_barrier, 0u, nullptr);
  for (const UploadPlan &plan : plans) {
    const VkBufferCopy copy{
        .srcOffset = plan.staging_offset,
        .dstOffset = plan.range.offset,
        .size = plan.range.bytes,
    };
    vkCmdCopyBuffer(adapter.command_buffer, staging.buffer.buffer,
                    plan.resident->buffer, 1u, &copy);
    const VkBufferMemoryBarrier target_barrier{
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = plan.resident->buffer,
        .offset = plan.range.offset,
        .size = plan.range.bytes,
    };
    vkCmdPipelineBarrier(adapter.command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                         1u, &target_barrier, 0u, nullptr);
  }
  return asynchronous
             ? SubmitVulkanTransfer(adapter, staging.buffer, std::move(targets))
             : SubmitVulkanTransferCommand(adapter);
}

#endif

} // namespace rund::node::accel::detail
