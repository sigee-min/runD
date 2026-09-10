#include "dispatch.hpp"

#include <cstring>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

void BindVulkanPipeline(const VkCommandBuffer command,
                        const VkPipelineBindPoint point,
                        const VkPipeline pipeline) noexcept {
  ::vkCmdBindPipeline(command, point, pipeline);
  VulkanDispatchCapture *const capture = vulkan_dispatch_capture;
  if (capture == nullptr || capture->command != command ||
      point != VK_PIPELINE_BIND_POINT_COMPUTE) {
    return;
  }
  capture->pipeline = pipeline;
  capture->layout = VK_NULL_HANDLE;
  capture->descriptor = VK_NULL_HANDLE;
  capture->has_push = false;
}

void BindVulkanDescriptors(
    const VkCommandBuffer command, const VkPipelineBindPoint point,
    const VkPipelineLayout layout, const std::uint32_t first_set,
    const std::uint32_t set_count, const VkDescriptorSet *const sets,
    const std::uint32_t dynamic_count,
    const std::uint32_t *const dynamic_offsets) noexcept {
  ::vkCmdBindDescriptorSets(command, point, layout, first_set, set_count, sets,
                            dynamic_count, dynamic_offsets);
  VulkanDispatchCapture *const capture = vulkan_dispatch_capture;
  if (capture == nullptr || capture->command != command ||
      point != VK_PIPELINE_BIND_POINT_COMPUTE) {
    return;
  }
  if (first_set != 0u || set_count != 1u || sets == nullptr ||
      sets[0] == VK_NULL_HANDLE || dynamic_count != 0u ||
      dynamic_offsets != nullptr) {
    capture->failed = true;
    return;
  }
  capture->layout = layout;
  capture->descriptor = sets[0];
}

void PushVulkanConstants(const VkCommandBuffer command,
                         const VkPipelineLayout layout,
                         const VkShaderStageFlags stages,
                         const std::uint32_t offset,
                         const std::uint32_t size,
                         const void *const data) noexcept {
  ::vkCmdPushConstants(command, layout, stages, offset, size, data);
  VulkanDispatchCapture *const capture = vulkan_dispatch_capture;
  if (capture == nullptr || capture->command != command) {
    return;
  }
  if (data == nullptr || offset > capture->push.size() ||
      size > capture->push.size() - offset) {
    capture->failed = true;
    return;
  }
  std::memcpy(capture->push.data() + offset, data, size);
  capture->push_stages = stages;
  capture->push_offset = offset;
  capture->push_size = size;
  capture->has_push = true;
}

void DispatchVulkan(const VkCommandBuffer command, const std::uint32_t x,
                    const std::uint32_t y,
                    const std::uint32_t z) noexcept {
  VulkanDispatchCapture *const capture = vulkan_dispatch_capture;
  if (capture == nullptr || capture->command != command) {
    WriteVulkanDispatchTimestamp(command);
    ::vkCmdDispatch(command, x, y, z);
    WriteVulkanDispatchTimestamp(command);
    return;
  }
  if (capture->mapped == nullptr || capture->original == nullptr ||
      capture->owners == nullptr || capture->arguments == VK_NULL_HANDLE ||
      capture->cursor >= capture->capacity) {
    capture->failed = true;
    return;
  }
  const std::size_t slot = capture->cursor++;
  const VkDispatchIndirectCommand value{.x = x, .y = y, .z = z};
  if (capture->replay) {
    if (capture->original[slot].x != value.x ||
        capture->original[slot].y != value.y ||
        capture->original[slot].z != value.z ||
        capture->owners[slot] != capture->owner) {
      capture->failed = true;
      return;
    }
  } else {
    capture->mapped[slot] = value;
    capture->original[slot] = value;
    capture->owners[slot] = capture->owner;
  }
  WriteVulkanDispatchTimestamp(command);
  ::vkCmdDispatchIndirect(
      command, capture->arguments,
      static_cast<VkDeviceSize>(slot * sizeof(VkDispatchIndirectCommand)));
  WriteVulkanDispatchTimestamp(command);
}

void DispatchVulkanIndirect(const VkCommandBuffer command,
                            const VkBuffer source,
                            const VkDeviceSize offset) noexcept {
  VulkanDispatchCapture *const capture = vulkan_dispatch_capture;
  if (capture == nullptr || capture->command != command) {
    WriteVulkanDispatchTimestamp(command);
    ::vkCmdDispatchIndirect(command, source, offset);
    WriteVulkanDispatchTimestamp(command);
    return;
  }
  if (capture->indirect == nullptr ||
      !capture->indirect(capture->context, *capture, command, source, offset)) {
    capture->failed = true;
  }
}

#endif // defined(RUND_NODE_HAVE_VULKAN_SDK)

} // namespace rund::node::accel::detail
