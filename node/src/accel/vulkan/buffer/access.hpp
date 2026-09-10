#pragma once

#include "../adapter/buffer.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanAdapter;

void DestroyVulkanBuffer(VulkanAdapter &adapter, VulkanBuffer &buffer);

void ReleaseVulkanBuffer(VulkanAdapter &adapter, VulkanBuffer &buffer);

[[nodiscard]] bool UploadVulkanBuffer(VulkanBuffer &buffer, const void *data,
                                      VkDeviceSize bytes);

[[nodiscard]] bool ClearVulkanBuffer(VulkanBuffer &buffer, VkDeviceSize bytes);

#endif // defined(RUND_NODE_HAVE_VULKAN_SDK)

} // namespace rund::node::accel::detail
