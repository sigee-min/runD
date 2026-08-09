#include "../control.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

#include "../../barrier.hpp"

#include <array>
#include <cstring>

namespace rund::node::accel::detail {

bool PublishVulkanPipelineControl(
    const VkCommandBuffer command,
    const VulkanPipelineControlResources &control) noexcept {
  if (command == VK_NULL_HANDLE || control.summary.buffer == VK_NULL_HANDLE) {
    return false;
  }
  std::array<VkBufferMemoryBarrier, 2u> visible{};
  std::uint32_t visible_count = 0u;
  visible[visible_count++] = VulkanBufferBarrier(
      control.summary, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
  if (control.profile.buffer != VK_NULL_HANDLE) {
    visible[visible_count++] = VulkanBufferBarrier(
        control.profile,
        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_HOST_READ_BIT);
  }
  const VkPipelineStageFlags source_stage =
      control.profile.buffer == VK_NULL_HANDLE
          ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
          : VK_PIPELINE_STAGE_TRANSFER_BIT |
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  vkCmdPipelineBarrier(command, source_stage, VK_PIPELINE_STAGE_HOST_BIT, 0u,
                       0u, nullptr, visible_count, visible.data(), 0u, nullptr);
  return true;
}

bool ReadVulkanPipelineControl(const VulkanPipelineControlResources &resources,
                               PreparedPipelineControl &control) noexcept {
  control = {};
  if (resources.summary.mapped == nullptr ||
      resources.summary.bytes < sizeof(control)) {
    return false;
  }
  std::memcpy(&control, resources.summary.mapped, sizeof(control));
  return true;
}

} // namespace rund::node::accel::detail

#endif
