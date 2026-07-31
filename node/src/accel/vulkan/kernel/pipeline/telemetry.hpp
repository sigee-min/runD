#pragma once

#include "state.hpp"

#include <span>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool PrepareVulkanTelemetry(VulkanPipeline &pipeline);
[[nodiscard]] bool
EncodeVulkanTelemetry(const VulkanPipeline &pipeline, VkCommandBuffer command,
                      std::span<const VulkanPipelineTelemetryRecord> telemetry,
                      bool visible = true) noexcept;
void ObserveVulkanProfile(VulkanPipeline &pipeline,
                          KernelResult &result) noexcept;

#endif

} // namespace rund::node::accel::detail
