#pragma once

#include "../control.hpp"
#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] VulkanMapControlPush VulkanMapControlParameters(
    const VulkanMapEncodeResources &, const rund::kernel::ComputeDispatchWindow &,
    std::uint32_t) noexcept;

void EncodeVulkanMapCheck(VkCommandBuffer,
                          const VulkanMapEncodeResources &);

void EncodeVulkanMapControl(
    VkCommandBuffer, const VulkanMapEncodeResources &,
    const rund::kernel::ComputeDispatchWindow &, std::uint32_t);

#endif

} // namespace rund::node::accel::detail
