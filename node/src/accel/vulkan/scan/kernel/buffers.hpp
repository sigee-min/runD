#pragma once

#include <accel/check.hpp>

#include "local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::AccelCheck
CreateVulkanScanScratch(VulkanAdapter &adapter,
                        const rund::kernel::ScanPlan &plan,
                        VulkanKernelScanResources &resources) {
  if (CreateVulkanBuffer(adapter, plan.block_count * plan.element_bytes,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, resources.totals,
                         nullptr, VulkanMemoryUse::Scratch) &&
      CreateVulkanStatus(adapter, sizeof(rund::kernel::u32),
                         resources.status)) {
    return rund::AccelCheck{true, "ok"};
  }
  return rund::AccelCheck{false, VulkanLastError(&adapter)};
}

} // namespace
#endif

} // namespace rund::node::accel::detail
