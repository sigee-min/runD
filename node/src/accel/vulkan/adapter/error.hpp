#pragma once

#include "state.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

void SetVulkanLastError(VulkanAdapter &adapter, const char *reason);

[[nodiscard]] constexpr const char *
VulkanFailureReason(const VkResult result,
                    const char *const fallback) noexcept {
  return result == VK_ERROR_DEVICE_LOST ? "compute_device_lost" : fallback;
}

[[nodiscard]] const char *VulkanLastError(void *context);

#endif // defined(RUND_NODE_HAVE_VULKAN_SDK)

} // namespace rund::node::accel::detail
