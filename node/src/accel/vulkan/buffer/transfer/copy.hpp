#pragma once

#include "../../adapter/error.hpp"

#include "../../command.hpp"
#include "../../scope.hpp"

#include <memory>
#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanCopy final {
  const VulkanBuffer *source = nullptr;
  VulkanBuffer *target = nullptr;
  VkDeviceSize source_offset = 0u;
  VkDeviceSize target_offset = 0u;
  VkDeviceSize bytes = 0u;
  VkPipelineStageFlags source_stage = 0u;
  VkAccessFlags source_access = 0u;
  VkPipelineStageFlags target_stage = 0u;
  VkAccessFlags target_access = 0u;
};

inline void EncodeVulkanBufferBarrier(
    const VkCommandBuffer command, const VulkanBuffer &buffer,
    const VkDeviceSize offset, const VkDeviceSize bytes,
    const VkPipelineStageFlags source_stage,
    const VkAccessFlags source_access,
    const VkPipelineStageFlags target_stage,
    const VkAccessFlags target_access) {
  const VkBufferMemoryBarrier barrier{
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .pNext = nullptr,
      .srcAccessMask = source_access,
      .dstAccessMask = target_access,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = buffer.buffer,
      .offset = offset,
      .size = bytes,
  };
  vkCmdPipelineBarrier(command, source_stage, target_stage, 0u, 0u, nullptr,
                       1u, &barrier, 0u, nullptr);
}

[[nodiscard]] inline bool EncodeVulkanBufferCopy(VulkanAdapter &adapter,
                                                 const VulkanCopy &copy) {
  const bool valid =
      copy.source != nullptr && copy.target != nullptr && copy.bytes != 0u &&
      copy.source->buffer != VK_NULL_HANDLE &&
      copy.target->buffer != VK_NULL_HANDLE &&
      copy.source_offset <= copy.source->bytes &&
      copy.bytes <= copy.source->bytes - copy.source_offset &&
      copy.target_offset <= copy.target->bytes &&
      copy.bytes <= copy.target->bytes - copy.target_offset;
  if (!valid) {
    SetVulkanLastError(adapter, "accel_vulkan_transfer_invalid");
    return false;
  }
  if (!BeginVulkanCommand(adapter)) {
    return false;
  }

  EncodeVulkanBufferBarrier(adapter.command_buffer, *copy.source,
                            copy.source_offset, copy.bytes, copy.source_stage,
                            copy.source_access,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_ACCESS_TRANSFER_READ_BIT);
  const VkBufferCopy region{
      .srcOffset = copy.source_offset,
      .dstOffset = copy.target_offset,
      .size = copy.bytes,
  };
  vkCmdCopyBuffer(adapter.command_buffer, copy.source->buffer,
                  copy.target->buffer, 1u, &region);
  EncodeVulkanBufferBarrier(adapter.command_buffer, *copy.target,
                            copy.target_offset, copy.bytes,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_ACCESS_TRANSFER_WRITE_BIT, copy.target_stage,
                            copy.target_access);
  return true;
}

[[nodiscard]] inline bool UploadVulkanCopy(
    VulkanAdapter &adapter, ScopedBuffer &&staging,
    VulkanBuffer &resident, const VkDeviceSize resident_offset,
    const VkDeviceSize bytes, std::shared_ptr<void> target = {}) {
  const VulkanCopy copy{
          .source = &staging.buffer,
          .target = &resident,
          .target_offset = resident_offset,
          .bytes = bytes,
          .source_stage = VK_PIPELINE_STAGE_HOST_BIT,
          .source_access = VK_ACCESS_HOST_WRITE_BIT,
          .target_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          .target_access =
              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
  };
  return EncodeVulkanBufferCopy(adapter, copy) &&
         SubmitVulkanTransfer(adapter, staging.buffer, std::move(target));
}

#endif

} // namespace rund::node::accel::detail
