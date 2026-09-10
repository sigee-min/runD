#pragma once

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool
MaterializeVulkanNumericPipelines(VulkanAdapter &adapter, const BoundStep &step,
                                  VulkanKernelImmutablePipelines &pipelines);

[[nodiscard]] bool
MaterializeVulkanScanPipelines(VulkanAdapter &adapter, const BoundStep &step,
                               VulkanKernelImmutablePipelines &pipelines);

[[nodiscard]] bool
MaterializeVulkanRangePipelines(VulkanAdapter &adapter, const BoundStep &step,
                                VulkanKernelImmutablePipelines &pipelines);

[[nodiscard]] bool
MaterializeVulkanCollectivePipelines(VulkanAdapter &adapter,
                                     const BoundStep &step,
                                     VulkanKernelImmutablePipelines &pipelines);

#endif

} // namespace rund::node::accel::detail
