#pragma once

#include "../local.hpp"

namespace rund::node::accel::detail::vulkan_residency_submit_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] rund::AccelCheck Invalid() noexcept;
[[nodiscard]] rund::AccelCheck Unavailable() noexcept;

void ServiceVulkanResidencyWindow(void *) noexcept;

#endif

} // namespace rund::node::accel::detail::vulkan_residency_submit_detail
