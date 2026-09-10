#pragma once

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
#include <vulkan/vulkan.h>
#endif

#include <cstdint>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanTimestampCapture final {
  VkQueryPool queries{VK_NULL_HANDLE};
  std::uint32_t cursor{};
  std::uint32_t capacity{};
  bool failed{};
};

extern thread_local VulkanTimestampCapture *vulkan_timestamp_capture;

class VulkanTimestampScope final {
public:
  explicit VulkanTimestampScope(VulkanTimestampCapture &capture) noexcept
      : previous_{vulkan_timestamp_capture} {
    vulkan_timestamp_capture = &capture;
  }

  ~VulkanTimestampScope() { vulkan_timestamp_capture = previous_; }

  VulkanTimestampScope(const VulkanTimestampScope &) = delete;
  VulkanTimestampScope &operator=(const VulkanTimestampScope &) = delete;

private:
  VulkanTimestampCapture *previous_{};
};

void WriteVulkanDispatchTimestamp(VkCommandBuffer command) noexcept;

#endif // defined(RUND_NODE_HAVE_VULKAN_SDK)

} // namespace rund::node::accel::detail
