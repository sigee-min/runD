#include "timestamp.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

thread_local VulkanTimestampCapture *vulkan_timestamp_capture = nullptr;

void WriteVulkanDispatchTimestamp(const VkCommandBuffer command) noexcept {
  VulkanTimestampCapture *const capture = vulkan_timestamp_capture;
  if (capture == nullptr || capture->queries == VK_NULL_HANDLE ||
      capture->cursor >= capture->capacity) {
    if (capture != nullptr) {
      capture->failed = true;
    }
    return;
  }
  ::vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        capture->queries, capture->cursor++);
}

#endif // defined(RUND_NODE_HAVE_VULKAN_SDK)

} // namespace rund::node::accel::detail
