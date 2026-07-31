#pragma once

#include "../adapter/api.hpp"
#include "stage.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] VulkanCollectivePipeline *AcquireVulkanScanPipeline(
    VulkanAdapter &adapter, const rund::kernel::ScanDesc &desc,
    rund::kernel::ComputeDomain domain, VulkanScanStage stage);

#endif

} // namespace rund::node::accel::detail
