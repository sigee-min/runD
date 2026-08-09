#include "../control.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

#include "../../barrier.hpp"

#include <cstring>
#include <limits>

namespace rund::node::accel::detail {

bool ResetVulkanPipelineProfile(
    const VkCommandBuffer command,
    const VulkanPipelineControlResources &control) noexcept {
  if (control.profile_step_count == 0u) {
    return control.profile.buffer == VK_NULL_HANDLE;
  }
  const VkDeviceSize field_bytes =
      static_cast<VkDeviceSize>(control.profile_step_count) *
      sizeof(std::uint64_t);
  const VkDeviceSize zero_bytes = 7u * field_bytes;
  const VkDeviceSize overflow_bytes = field_bytes;
  if (command == VK_NULL_HANDLE || control.profile.buffer == VK_NULL_HANDLE ||
      control.profile.bytes < zero_bytes + overflow_bytes) {
    return false;
  }
  vkCmdFillBuffer(command, control.profile.buffer, 0u, zero_bytes, 0u);
  vkCmdFillBuffer(command, control.profile.buffer, zero_bytes, overflow_bytes,
                  std::numeric_limits<std::uint32_t>::max());
  const VkBufferMemoryBarrier visible = VulkanBufferBarrier(
      control.profile, VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                       1u, &visible, 0u, nullptr);
  return true;
}

bool ReadVulkanPipelineProfile(
    const VulkanPipelineControlResources &resources,
    const std::span<PreparedPipelineStepControl> controls) noexcept {
  const std::size_t step_count = resources.profile_step_count;
  const std::uint64_t required_bytes =
      static_cast<std::uint64_t>(step_count) * PreparedPipelineStepControlBytes;
  if (step_count == 0u) {
    return resources.profile.buffer == VK_NULL_HANDLE && controls.empty();
  }
  if (resources.profile.mapped == nullptr ||
      resources.profile.bytes < required_bytes ||
      controls.size() < step_count) {
    return false;
  }
  const auto *const bytes =
      static_cast<const std::uint8_t *>(resources.profile.mapped);
  const auto read = [bytes, step_count](const std::size_t field,
                                        const std::size_t step) noexcept {
    std::uint64_t value{};
    const std::size_t offset =
        (field * step_count + step) * sizeof(std::uint64_t);
    std::memcpy(&value, bytes + offset, sizeof(value));
    return value;
  };
  for (std::size_t step = 0u; step < step_count; ++step) {
    controls[step] = PreparedPipelineStepControl{
        .generated_item_count = read(0u, step),
        .generated_capacity = read(1u, step),
        .indirect_dispatch_count = read(2u, step),
        .indirect_work_item_count = read(3u, step),
        .iteration_count = read(4u, step),
        .skipped_iteration_count = read(5u, step),
        .conflict_count = read(6u, step),
        .overflow_ordinal = read(7u, step),
    };
  }
  return true;
}

} // namespace rund::node::accel::detail

#endif
