#pragma once

#include <kernel/program/compute/backend.hpp>

#include <string_view>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] rund::kernel::ComputePlan
VulkanPipelineControlPlan(std::string_view source) noexcept;

#endif

} // namespace rund::node::accel::detail
