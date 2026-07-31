#pragma once

#include <kernel/program/compute/model.hpp>

#include <string>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] const std::string &VulkanTelemetrySource();
[[nodiscard]] const std::string &VulkanProfileSource();
[[nodiscard]] rund::kernel::ComputePlan VulkanTelemetryPlan() noexcept;
[[nodiscard]] rund::kernel::ComputePlan VulkanProfilePlan() noexcept;

#endif

} // namespace rund::node::accel::detail
