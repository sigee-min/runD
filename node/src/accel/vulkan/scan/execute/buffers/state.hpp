#pragma once

#include <accel/check.hpp>

#include "../../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

struct VulkanScanDirectState {
  VulkanBuffer totals{};
  VulkanStatus status{};
  std::shared_ptr<void> resources{};
};

void CleanupVulkanScanDirect(VulkanAdapter &adapter,
                             VulkanScanDirectState &state) {
  state.resources.reset();
  ReleaseVulkanBuffer(adapter, state.totals);
  ReleaseVulkanStatus(adapter, state.status);
}

[[nodiscard]] rund::AccelCheck ValidateVulkanScanDirect(
    VulkanAdapter &adapter, const rund::kernel::ScanDesc &desc,
    const rund::kernel::ScanPlan &plan, const VulkanBuffer &input,
    const VulkanBuffer &output) {
  SetVulkanLastError(adapter, "ok");
  if (adapter.device == VK_NULL_HANDLE ||
      adapter.compute_queue == VK_NULL_HANDLE ||
      input.buffer == VK_NULL_HANDLE || output.buffer == VK_NULL_HANDLE ||
      !ScanShapeOk(desc, plan)) {
    SetVulkanLastError(adapter, "compute_scan_invalid");
    return rund::AccelCheck{false, "compute_scan_invalid"};
  }
  return rund::AccelCheck{true, "ok"};
}

} // namespace
#endif

} // namespace rund::node::accel::detail
