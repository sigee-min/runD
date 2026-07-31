#pragma once

#include <accel/check.hpp>

#include "../local.hpp"

#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] bool VulkanScanU32AbiOk(const rund::kernel::ScanPlan &plan) {
  const auto u32_max =
      static_cast<rund::kernel::u64>(std::numeric_limits<std::uint32_t>::max());
  return plan.block_count <= u32_max && plan.element_count <= u32_max &&
         plan.block_size <= u32_max;
}

[[nodiscard]] rund::AccelCheck ValidateVulkanScanResources(
    VulkanAdapter &adapter, const rund::kernel::ScanDesc &desc,
    const rund::kernel::ScanPlan &plan, const VulkanBuffer &input,
    const VulkanBuffer &output, const VulkanBuffer &totals,
    const VulkanStatus &status) {
  if (adapter.device == VK_NULL_HANDLE || input.buffer == VK_NULL_HANDLE ||
      output.buffer == VK_NULL_HANDLE || totals.buffer == VK_NULL_HANDLE ||
      status.device.buffer == VK_NULL_HANDLE || !ScanShapeOk(desc, plan) ||
      !VulkanScanU32AbiOk(plan)) {
    SetVulkanLastError(adapter, "compute_scan_invalid");
    return rund::AccelCheck{false, "compute_scan_invalid"};
  }
  const rund::kernel::u64 scan_bytes = plan.element_count * plan.element_bytes;
  const rund::kernel::u64 totals_bytes = plan.block_count * plan.element_bytes;
  constexpr rund::kernel::u64 status_bytes = sizeof(rund::kernel::u32);
  if (input.bytes < scan_bytes || output.bytes < scan_bytes ||
      totals.bytes < totals_bytes || status.device.bytes < status_bytes) {
    SetVulkanLastError(adapter, "compute_scan_invalid");
    return rund::AccelCheck{false, "compute_scan_invalid"};
  }
  return rund::AccelCheck{true, "ok"};
}

} // namespace
#endif

} // namespace rund::node::accel::detail
