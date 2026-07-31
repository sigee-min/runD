#include "local.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] std::uint64_t
latest_sequence(const VulkanAdapter &adapter) noexcept {
  std::uint64_t sequence = 0u;
  for (std::size_t slot = 0u; slot < adapter.command_ring.phases.size();
       ++slot) {
    if (adapter.command_ring.phases[slot] == VulkanCommandPhase::Submitted) {
      sequence = std::max(sequence, adapter.command_ring.sequences[slot]);
    }
  }
  return sequence;
}

void wait_sequence(VulkanAdapter &adapter, std::unique_lock<std::mutex> &lock,
                   const std::uint64_t sequence) {
  if (sequence == 0u) {
    return;
  }
  adapter.command_cv.wait(lock, [&adapter, sequence] {
    for (std::size_t slot = 0u; slot < adapter.command_ring.phases.size();
         ++slot) {
      if (adapter.command_ring.phases[slot] == VulkanCommandPhase::Submitted &&
          adapter.command_ring.sequences[slot] <= sequence) {
        return false;
      }
    }
    return true;
  });
}

[[nodiscard]] bool overlaps(const std::span<const UploadPlan> plans) {
  if (plans.size() < 2u) {
    return false;
  }
  std::vector<const UploadPlan *> ordered;
  ordered.reserve(plans.size());
  for (const UploadPlan &plan : plans) {
    ordered.push_back(&plan);
  }
  std::sort(ordered.begin(), ordered.end(),
            [](const UploadPlan *const lhs, const UploadPlan *const rhs) {
              if (lhs->resident != rhs->resident) {
                return std::less<const VulkanBuffer *>{}(lhs->resident,
                                                         rhs->resident);
              }
              return lhs->range.offset < rhs->range.offset;
            });
  const VulkanBuffer *resident = nullptr;
  VkDeviceSize end = 0u;
  for (const UploadPlan *const plan : ordered) {
    if (plan->resident != resident) {
      resident = plan->resident;
      end = plan->range.offset + plan->range.bytes;
      continue;
    }
    if (plan->range.offset < end) {
      return true;
    }
    end = std::max(end, plan->range.offset + plan->range.bytes);
  }
  return false;
}

[[nodiscard]] bool encode_download(VulkanAdapter &adapter,
                                   const std::span<const DownloadPlan> plans,
                                   VulkanBuffer &staging,
                                   const VkDeviceSize staging_bytes) {
  std::vector<VkBufferMemoryBarrier> source_barriers;
  source_barriers.reserve(plans.size());
  for (const DownloadPlan &plan : plans) {
    source_barriers.push_back(VkBufferMemoryBarrier{
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
    });
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
  return SubmitVulkanCommand(adapter, false);
}

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
  return SubmitVulkanCommand(adapter, false);
}

[[nodiscard]] bool
encode_upload(VulkanAdapter &adapter, const std::span<const UploadPlan> plans,
              ScopedBuffer &staging, const VkDeviceSize staging_bytes,
              const bool asynchronous, std::shared_ptr<void> targets = {}) {
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
             : SubmitVulkanCommand(adapter, false);
}

#endif

} // namespace rund::node::accel::detail
