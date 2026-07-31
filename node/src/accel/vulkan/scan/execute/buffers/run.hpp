#pragma once

#include <accel/check.hpp>

#include "finish.hpp"

namespace rund::node::accel::detail {

rund::AccelCheck ExecuteVulkanScanBuffers(
    VulkanAdapter &adapter, const rund::kernel::ScanDesc &desc,
    const rund::kernel::ScanPlan &plan,
    const rund::kernel::ComputeDomain domain, const VulkanBuffer &input,
    const VulkanBuffer &output, const bool record_dispatches,
    const VulkanBuffer &logical_count) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const rund::AccelCheck valid =
      ValidateVulkanScanDirect(adapter, desc, plan, input, output);
  if (!valid.ok) {
    return valid;
  }
  VulkanScanDirectState state{};
  rund::AccelCheck check = CreateVulkanScanDirectScratch(adapter, plan, state);
  if (!check.ok) {
    return check;
  }
  check = PrepareVulkanScanDirect(adapter, desc, plan, domain, input, output,
                                  logical_count, state);
  if (!check.ok) {
    CleanupVulkanScanDirect(adapter, state);
    return check;
  }
  check = SubmitVulkanScanDirect(adapter, state);
  if (!check.ok) {
    CleanupVulkanScanDirect(adapter, state);
    return check;
  }
  return FinishVulkanScanDirect(adapter, state, record_dispatches);
#else
  (void)adapter;
  (void)desc;
  (void)plan;
  (void)domain;
  (void)input;
  (void)output;
  (void)record_dispatches;
  (void)logical_count;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
