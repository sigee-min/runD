#pragma once

#include <accel/check.hpp>

#include "scratch.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::AccelCheck PrepareVulkanScanDirect(
    VulkanAdapter &adapter, const rund::kernel::ScanDesc &desc,
    const rund::kernel::ScanPlan &plan,
    const rund::kernel::ComputeDomain domain, const VulkanBuffer &input,
    const VulkanBuffer &output, const VulkanBuffer &logical_count,
    VulkanScanDirectState &state) {
  return PrepareVulkanScanBuffers(adapter, desc, plan, domain, input, output,
                                  logical_count, state.totals, state.status,
                                  state.resources);
}

} // namespace
#endif

} // namespace rund::node::accel::detail
