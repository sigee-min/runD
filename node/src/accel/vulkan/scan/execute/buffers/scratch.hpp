#pragma once

#include <accel/check.hpp>

#include "state.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::AccelCheck
CreateVulkanScanDirectScratch(VulkanAdapter &adapter,
                              const rund::kernel::ScanPlan &plan,
                              VulkanScanDirectState &state) {
  if (!CreateVulkanBuffer(adapter, plan.block_count * plan.element_bytes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, state.totals,
                          nullptr, VulkanMemoryUse::Scratch) ||
      !CreateVulkanStatus(adapter, sizeof(rund::kernel::u32), state.status)) {
    CleanupVulkanScanDirect(adapter, state);
    return rund::AccelCheck{false, VulkanLastError(&adapter)};
  }
  return rund::AccelCheck{true, "ok"};
}

} // namespace
#endif

} // namespace rund::node::accel::detail
