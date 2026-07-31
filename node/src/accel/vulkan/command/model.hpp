#pragma once

#include <cstdint>

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
#include <vulkan/vulkan.h>
#endif

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

enum class CommandKind : std::uint8_t {
  OneShotPrimary,
  ImmutableSecondary,
  ReusablePrimary,
};

struct VulkanCommand final {
  VkCommandPool pool{VK_NULL_HANDLE};
  VkCommandBuffer buffer{VK_NULL_HANDLE};
  VkFence fence{VK_NULL_HANDLE};
};

#endif

} // namespace rund::node::accel::detail
