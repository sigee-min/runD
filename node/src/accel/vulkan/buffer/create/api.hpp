#pragma once

#include "../../adapter/buffer.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
[[nodiscard]] bool CreateFreshVulkanBuffer(
    VulkanAdapter &adapter, VkDeviceSize bytes, VkBufferUsageFlags usage,
    VulkanMemoryUse memory_use, VulkanBuffer &buffer);
#endif

} // namespace rund::node::accel::detail
