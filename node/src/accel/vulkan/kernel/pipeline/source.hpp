#pragma once

#include <kernel/program/compute/model.hpp>

#include <string>
#include <string_view>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] std::string_view VulkanTelemetrySourceText() noexcept;
[[nodiscard]] std::string VulkanProfileSource();
[[nodiscard]] std::uint64_t VulkanProfileSourceBytes() noexcept;
[[nodiscard]] rund::kernel::ComputePlan VulkanTelemetryPlan() noexcept;
[[nodiscard]] rund::kernel::ComputePlan VulkanProfilePlan() noexcept;

#endif

} // namespace rund::node::accel::detail
