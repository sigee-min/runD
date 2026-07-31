#include "status.hpp"

#include "adapter/api.hpp"
#include "barrier.hpp"
#include "descriptor/binding.hpp"

#include <algorithm>
#include <array>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {
inline constexpr VkDeviceSize kVulkanStatusReadLimit =
    4u * sizeof(rund::kernel::u32);
inline constexpr std::size_t kVulkanStatusOutputCapacity = 2u;

[[nodiscard]] bool CopyVulkanStatus(const VkCommandBuffer command,
                                    const VulkanStatus &status) noexcept {
  if (command == VK_NULL_HANDLE || status.device.buffer == VK_NULL_HANDLE ||
      status.readback.buffer == VK_NULL_HANDLE || status.read_bytes == 0u ||
      status.read_bytes > status.device.bytes ||
      status.read_bytes > status.readback.bytes ||
      status.readback.mapped == nullptr ||
      (status.device.usage & VK_BUFFER_USAGE_TRANSFER_SRC_BIT) == 0u ||
      (status.readback.usage & VK_BUFFER_USAGE_TRANSFER_DST_BIT) == 0u) {
    return false;
  }
  const VkBufferCopy copy{
      .srcOffset = 0u, .dstOffset = 0u, .size = status.read_bytes};
  vkCmdCopyBuffer(command, status.device.buffer, status.readback.buffer, 1u,
                  &copy);
  const VkBufferMemoryBarrier visible = VulkanBufferBarrier(
      status.readback, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0u, 0u, nullptr, 1u,
                       &visible, 0u, nullptr);
  return true;
}
} // namespace

bool CreateVulkanStatus(VulkanAdapter &adapter, const VkDeviceSize bytes,
                        VulkanStatus &status,
                        const KernelPreparationMode mode) {
  status = {};
  status.pipeline = IsPipelinePrivatePreparation(mode);
  const VkDeviceSize read_bytes = std::min(bytes, kVulkanStatusReadLimit);
  if (bytes < sizeof(rund::kernel::u32) || (bytes & 3u) != 0u ||
      !CreateVulkanBuffer(adapter, bytes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          status.device, nullptr, VulkanMemoryUse::Device) ||
      (!status.pipeline && !CreateVulkanBuffer(adapter, read_bytes,
                                               VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                               status.readback))) {
    ReleaseVulkanStatus(adapter, status);
    return false;
  }
  status.read_bytes = status.pipeline ? 0u : read_bytes;
  return true;
}

void ReleaseVulkanStatus(VulkanAdapter &adapter,
                         VulkanStatus &status) noexcept {
  ReleaseVulkanBuffer(adapter, status.device);
  ReleaseVulkanBuffer(adapter, status.readback);
  status = {};
}

bool ResetVulkanStatus(const VkCommandBuffer command,
                       const VulkanBuffer &status, const VkDeviceSize bytes,
                       const std::uint32_t value) noexcept {
  return ResetVulkanStatus(command, VulkanStorageBindingFor(status), bytes,
                           value);
}

bool ResetVulkanStatus(const VkCommandBuffer command,
                       const VulkanStorageBinding &status,
                       const VkDeviceSize bytes,
                       const std::uint32_t value) noexcept {
  if (command == VK_NULL_HANDLE || status.buffer == nullptr ||
      status.buffer->buffer == VK_NULL_HANDLE || bytes == 0u ||
      (status.offset & 3u) != 0u || (bytes & 3u) != 0u ||
      bytes > status.range ||
      (status.buffer->usage & VK_BUFFER_USAGE_TRANSFER_DST_BIT) == 0u) {
    return false;
  }
  // Vulkan command order alone does not order an earlier compute read before a
  // later transfer fill.  This WAR edge is required when Pipeline programs
  // reuse one public status range: canonicalization must observe the first
  // value before the next program clears it.
  const VkBufferMemoryBarrier writable = VulkanBufferBarrier(
      *status.buffer, status.offset, bytes,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
      VK_ACCESS_TRANSFER_WRITE_BIT);
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0u, 0u, nullptr, 1u,
                       &writable, 0u, nullptr);
  vkCmdFillBuffer(command, status.buffer->buffer, status.offset, bytes, value);
  const VkBufferMemoryBarrier ready = VulkanBufferBarrier(
      *status.buffer, status.offset, bytes, VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                       1u, &ready, 0u, nullptr);
  return true;
}

bool ResetVulkanStatus(const VkCommandBuffer command,
                       const VulkanStatus &status, const VkDeviceSize bytes,
                       const std::uint32_t value) noexcept {
  return ResetVulkanStatus(command, status.device, bytes, value);
}

bool FinishVulkanStatus(
    const VkCommandBuffer command, const VulkanStatus &status,
    const std::span<const VulkanBuffer *const> outputs) noexcept {
  if (command == VK_NULL_HANDLE ||
      outputs.size() > kVulkanStatusOutputCapacity ||
      status.device.buffer == VK_NULL_HANDLE) {
    return false;
  }
  std::array<VkBufferMemoryBarrier, kVulkanStatusOutputCapacity + 1u>
      barriers{};
  for (std::size_t index = 0u; index < outputs.size(); ++index) {
    if (outputs[index] == nullptr || outputs[index]->buffer == VK_NULL_HANDLE) {
      return false;
    }
    barriers[index] = VulkanDeviceOutputBarrier(*outputs[index]);
  }
  const std::size_t barrier_count =
      outputs.size() + static_cast<std::size_t>(!status.pipeline);
  if (barrier_count == 0u) {
    return true;
  }
  if (!status.pipeline) {
    barriers[outputs.size()] = VulkanBufferBarrier(
        status.device, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
  }
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       kVulkanDeviceOutputStage, 0u, 0u, nullptr,
                       static_cast<std::uint32_t>(barrier_count),
                       barriers.data(), 0u, nullptr);
  return status.pipeline || CopyVulkanStatus(command, status);
}

const rund::kernel::u32 *
VulkanStatusValue(const VulkanStatus &status) noexcept {
  return static_cast<const rund::kernel::u32 *>(status.readback.mapped);
}
#endif

} // namespace rund::node::accel::detail
