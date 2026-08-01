#include "local.hpp"

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
             : SubmitVulkanCommand(adapter, false);
}

BackendCopy
CopyVulkanResidentBuffers(const rund::AccelDevice &pick,
                          const std::span<const CopyRoute> requests) {
  if (!VulkanPickOwnsAdapter(pick) || requests.empty()) {
    return {};
  }
  struct CopyPlan final {
    VulkanBuffer *source = nullptr;
    VulkanBuffer *target = nullptr;
    std::shared_ptr<void> source_storage;
    std::shared_ptr<void> target_storage;
    VkDeviceSize source_offset = 0u;
    VkDeviceSize target_offset = 0u;
    VkDeviceSize bytes = 0u;
  };
  auto *const adapter = static_cast<VulkanAdapter *>(pick.backend.context);
  std::unique_lock<std::mutex> lock{adapter->mutex};
  try {
    std::vector<CopyPlan> plans;
    plans.reserve(requests.size());
    {
      VulkanResidentState &resident = VulkanResidents(*adapter);
      std::lock_guard resident_lock{resident.mutex};
      for (const CopyRoute &request : requests) {
        VulkanResidentBufferResult source = ResolveVulkanResidentBuffer(
            resident, request.source, request.source_handle,
            "accel_buffer_unavailable");
        VulkanResidentBufferResult target = ResolveVulkanResidentBuffer(
            resident, request.target, request.target_handle,
            "accel_buffer_unavailable");
        if (!source.check.ok || !target.check.ok ||
            source.device_buffer == nullptr ||
            target.device_buffer == nullptr ||
            request.source_offset > request.source.bytes ||
            request.bytes > request.source.bytes - request.source_offset ||
            request.target_offset > request.target.bytes ||
            request.bytes > request.target.bytes - request.target_offset ||
            request.source.offset_bytes >
                std::numeric_limits<std::uint64_t>::max() -
                    request.source_offset ||
            request.target.offset_bytes >
                std::numeric_limits<std::uint64_t>::max() -
                    request.target_offset) {
          return BackendCopy{
              .check = {false, !source.check.ok ? source.check.reason
                               : !target.check.ok
                                   ? target.check.reason
                                   : "accel_buffer_copy_overflow"}};
        }
        if (request.bytes == 0u) {
          continue;
        }
        const std::uint64_t source_offset =
            request.source.offset_bytes + request.source_offset;
        const std::uint64_t target_offset =
            request.target.offset_bytes + request.target_offset;
        if (((source_offset | target_offset | request.bytes) & 3u) != 0u) {
          return BackendCopy{.check = {false, "accel_buffer_copy_alignment"}};
        }
        plans.push_back(CopyPlan{
            .source = source.device_buffer,
            .target = target.device_buffer,
            .source_storage = std::move(source.storage),
            .target_storage = std::move(target.storage),
            .source_offset = static_cast<VkDeviceSize>(source_offset),
            .target_offset = static_cast<VkDeviceSize>(target_offset),
            .bytes = static_cast<VkDeviceSize>(request.bytes),
        });
      }
    }
    const auto overlaps = [](const VkDeviceSize left_offset,
                             const VkDeviceSize left_bytes,
                             const VkDeviceSize right_offset,
                             const VkDeviceSize right_bytes) noexcept {
      return left_offset < right_offset + right_bytes &&
             right_offset < left_offset + left_bytes;
    };
    for (std::size_t target_index = 0u; target_index < plans.size();
         ++target_index) {
      const CopyPlan &target = plans[target_index];
      for (std::size_t source_index = 0u; source_index < plans.size();
           ++source_index) {
        const CopyPlan &source = plans[source_index];
        if (target.target->buffer == source.source->buffer &&
            overlaps(target.target_offset, target.bytes, source.source_offset,
                     source.bytes)) {
          return BackendCopy{.check = {false, "accel_buffer_copy_overlap"}};
        }
        if (source_index != target_index &&
            target.target->buffer == source.target->buffer &&
            overlaps(target.target_offset, target.bytes, source.target_offset,
                     source.bytes)) {
          return BackendCopy{.check = {false, "accel_buffer_copy_overlap"}};
        }
      }
    }
    if (plans.empty()) {
      return BackendCopy{.check = {true, "ok"}};
    }
    WaitForVulkanCommandSlot(*adapter, lock);
    if (!BeginVulkanCommand(*adapter)) {
      return BackendCopy{.check = {false, VulkanLastError(adapter)}};
    }
    std::vector<VkBufferMemoryBarrier> source_barriers;
    std::vector<VkBufferMemoryBarrier> target_barriers;
    source_barriers.reserve(plans.size() * 2u);
    target_barriers.reserve(plans.size());
    for (const CopyPlan &plan : plans) {
      source_barriers.push_back(VkBufferMemoryBarrier{
          .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
          .pNext = nullptr,
          .srcAccessMask =
              VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
          .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .buffer = plan.source->buffer,
          .offset = plan.source_offset,
          .size = plan.bytes,
      });
      source_barriers.push_back(VkBufferMemoryBarrier{
          .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
          .pNext = nullptr,
          .srcAccessMask =
              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
              VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
          .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .buffer = plan.target->buffer,
          .offset = plan.target_offset,
          .size = plan.bytes,
      });
      target_barriers.push_back(VkBufferMemoryBarrier{
          .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
          .pNext = nullptr,
          .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
          .dstAccessMask =
              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .buffer = plan.target->buffer,
          .offset = plan.target_offset,
          .size = plan.bytes,
      });
    }
    vkCmdPipelineBarrier(adapter->command_buffer,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0u, 0u, nullptr,
                         static_cast<std::uint32_t>(source_barriers.size()),
                         source_barriers.data(), 0u, nullptr);
    for (const CopyPlan &plan : plans) {
      const VkBufferCopy copy{.srcOffset = plan.source_offset,
                              .dstOffset = plan.target_offset,
                              .size = plan.bytes};
      vkCmdCopyBuffer(adapter->command_buffer, plan.source->buffer,
                      plan.target->buffer, 1u, &copy);
    }
    vkCmdPipelineBarrier(adapter->command_buffer,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                         static_cast<std::uint32_t>(target_barriers.size()),
                         target_barriers.data(), 0u, nullptr);
    const bool copied = SubmitVulkanCommand(*adapter, false);
    return BackendCopy{
        .check = copied ? rund::AccelCheck{true, "ok"}
                        : rund::AccelCheck{false, VulkanLastError(adapter)},
        .command_submits = 1u,
    };
  } catch (const std::bad_alloc &) {
    return BackendCopy{.check = {false, "accel_buffer_unavailable"}};
  }
}

#endif

} // namespace rund::node::accel::detail
