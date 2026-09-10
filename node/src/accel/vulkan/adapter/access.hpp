#pragma once

#include <accel/device.hpp>

#include "state.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool VulkanPickOwnsAdapter(const rund::AccelDevice &pick) noexcept;

[[nodiscard]] VulkanAdapter *
CheckedVulkanAdapter(const rund::AccelDevice &pick) noexcept;

#endif // defined(RUND_NODE_HAVE_VULKAN_SDK)

} // namespace rund::node::accel::detail
