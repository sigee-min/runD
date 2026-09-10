#pragma once

#include "../../command/capture.hpp"
#include "../window.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] rund::AccelCheck PrepareVulkanWindowResources(
    VulkanAdapter &adapter, std::span<const BackendBatchEntry> entries,
    std::uint64_t dispatch_capacity, std::uint64_t gate_capacity,
    const PreparedPipelineStatusLayout &status, const VulkanBuffer &control,
    VulkanWindowResources &resources);

[[nodiscard]] rund::AccelCheck
PrepareVulkanWindowDescriptors(VulkanAdapter &adapter,
                               const VulkanBuffer &control,
                               VulkanWindowResources &resources);

[[nodiscard]] bool EncodeVulkanWindowIndirect(void *context,
                                              VulkanDispatchCapture &capture,
                                              VkCommandBuffer command,
                                              VkBuffer source,
                                              VkDeviceSize offset) noexcept;

#endif

} // namespace rund::node::accel::detail
