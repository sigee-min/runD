#include "../../../../../backend/result.hpp"

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

[[nodiscard]] bool encode_download(VulkanAdapter &adapter,
                                   const std::span<const DownloadPlan> plans,
                                   VulkanBuffer &staging,
                                   const VkDeviceSize staging_bytes,
                                   bool *const submitted) {
  std::array<VkBufferMemoryBarrier, kInlineTransferCapacity> inline_barriers{};
  std::vector<VkBufferMemoryBarrier> overflow_barriers;
  std::span<VkBufferMemoryBarrier> source_barriers{};
  if (plans.size() <= inline_barriers.size()) {
    source_barriers = std::span{inline_barriers}.first(plans.size());
  } else {
    overflow_barriers.resize(plans.size());
    source_barriers = overflow_barriers;
  }
  for (std::size_t index = 0u; index < plans.size(); ++index) {
    const DownloadPlan &plan = plans[index];
    source_barriers[index] = VkBufferMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask =
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = plan.resident->buffer,
        .offset = plan.range.offset,
        .size = plan.range.bytes,
    };
  }
  if (!BeginVulkanCommand(adapter)) {
    return false;
  }
  vkCmdPipelineBarrier(adapter.command_buffer,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0u, 0u, nullptr,
                       static_cast<std::uint32_t>(source_barriers.size()),
                       source_barriers.data(), 0u, nullptr);
  for (const DownloadPlan &plan : plans) {
    const VkBufferCopy copy{
        .srcOffset = plan.range.offset,
        .dstOffset = plan.staging_offset,
        .size = plan.range.bytes,
    };
    vkCmdCopyBuffer(adapter.command_buffer, plan.resident->buffer,
                    staging.buffer, 1u, &copy);
  }
  const VkBufferMemoryBarrier target_barrier{
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .pNext = nullptr,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = staging.buffer,
      .offset = 0u,
      .size = staging_bytes,
  };
  vkCmdPipelineBarrier(adapter.command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0u, 0u, nullptr, 1u,
                       &target_barrier, 0u, nullptr);
  return SubmitVulkanTransferCommand(adapter, submitted);
}

#endif

} // namespace rund::node::accel::detail
