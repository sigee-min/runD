#pragma once

#include "../adapter/buffer.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool
CreateVulkanBuffer(VulkanAdapter &adapter, VkDeviceSize bytes,
                   VkBufferUsageFlags usage, VulkanBuffer &buffer,
                   bool *reused = nullptr,
                   VulkanMemoryUse memory_use = VulkanMemoryUse::Staging,
                   std::uint64_t exact_storage_bytes = 0u);

#endif

} // namespace rund::node::accel::detail
