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

[[nodiscard]] bool encode_preserve(VulkanAdapter &adapter,
                                   const std::span<const UploadPlan> plans,
                                   VulkanBuffer &staging,
                                   const VkDeviceSize staging_bytes) {
  std::size_t region_count = 0u;
  for (const UploadPlan &plan : plans) {
    region_count += plan.preservation.region_count;
  }
  if (region_count == 0u) {
    return true;
  }
  std::vector<VkBufferMemoryBarrier> source_barriers;
  source_barriers.reserve(region_count);
  if (!BeginVulkanCommand(adapter)) {
    return false;
  }
  for (const UploadPlan &plan : plans) {
    for (std::uint32_t index = 0u; index < plan.preservation.region_count;
         ++index) {
      const VkBufferCopy &preserved = plan.preservation.regions[index];
      source_barriers.push_back(VkBufferMemoryBarrier{
          .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
          .pNext = nullptr,
          .srcAccessMask =
              VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
          .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .buffer = plan.resident->buffer,
          .offset = preserved.srcOffset,
          .size = preserved.size,
      });
    }
  }
  vkCmdPipelineBarrier(adapter.command_buffer,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0u, 0u, nullptr,
                       static_cast<std::uint32_t>(source_barriers.size()),
                       source_barriers.data(), 0u, nullptr);
  for (const UploadPlan &plan : plans) {
    for (std::uint32_t index = 0u; index < plan.preservation.region_count;
         ++index) {
      VkBufferCopy copy = plan.preservation.regions[index];
      copy.dstOffset += plan.staging_offset;
      vkCmdCopyBuffer(adapter.command_buffer, plan.resident->buffer,
                      staging.buffer, 1u, &copy);
    }
  }
  const VkBufferMemoryBarrier target_barrier{
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .pNext = nullptr,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_HOST_WRITE_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = staging.buffer,
      .offset = 0u,
      .size = staging_bytes,
  };
  vkCmdPipelineBarrier(adapter.command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0u, 0u, nullptr, 1u,
                       &target_barrier, 0u, nullptr);
  return SubmitVulkanTransferCommand(adapter);
}

#endif

} // namespace rund::node::accel::detail
