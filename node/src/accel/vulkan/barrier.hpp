#pragma once

#include "adapter/buffer.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] inline VkBufferMemoryBarrier
VulkanBufferBarrier(const VulkanBuffer &buffer, const VkDeviceSize offset,
                    const VkDeviceSize bytes, const VkAccessFlags src,
                    const VkAccessFlags dst) noexcept {
  return VkBufferMemoryBarrier{
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .pNext = nullptr,
      .srcAccessMask = src,
      .dstAccessMask = dst,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = buffer.buffer,
      .offset = offset,
      .size = bytes,
  };
}

[[nodiscard]] inline VkBufferMemoryBarrier
VulkanBufferBarrier(const VulkanBuffer &buffer, const VkAccessFlags src,
                    const VkAccessFlags dst) noexcept {
  return VulkanBufferBarrier(buffer, buffer.offset, buffer.bytes, src, dst);
}

[[nodiscard]] inline VkBufferMemoryBarrier
VulkanDeviceOutputBarrier(const VulkanBuffer &buffer) noexcept {
  return VulkanBufferBarrier(buffer, VK_ACCESS_SHADER_WRITE_BIT,
                             VK_ACCESS_TRANSFER_READ_BIT |
                                 VK_ACCESS_SHADER_READ_BIT |
                                 VK_ACCESS_SHADER_WRITE_BIT);
}

inline constexpr VkPipelineStageFlags kVulkanDeviceOutputStage =
    VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

#endif

} // namespace rund::node::accel::detail
