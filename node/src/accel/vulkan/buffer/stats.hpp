#pragma once

#include <accel/device.hpp>
#include <accel/runtime.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] rund::RuntimeStats
ReadVulkanRuntimeStats(const rund::AccelDevice &pick);

void ResetVulkanRuntimeStats(const rund::AccelDevice &pick);

#endif // defined(RUND_NODE_HAVE_VULKAN_SDK)

} // namespace rund::node::accel::detail
