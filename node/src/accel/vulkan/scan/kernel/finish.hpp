#pragma once

#include <accel/check.hpp>

#include <rund/counter.hpp>
#include "../../kernel/ops/status.hpp"
#include "../local.hpp"
#include "local.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelCheck DescribeVulkanScanPipelineStatus(
    const std::shared_ptr<void> &resources,
    VulkanPipelineStatusSource &source) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const scan = static_cast<VulkanKernelScanResources *>(resources.get());
  constexpr std::array mapping{
      VulkanPipelineStatusMapping{1u,
                                  rund::compute::Reason::ScanSumOverflow},
      VulkanPipelineStatusMapping{2u,
                                  rund::compute::Reason::BoundedCountInvalid},
      VulkanPipelineStatusMapping{3u,
                                  rund::compute::Reason::BoundedCountInvalid},
  };
  return scan == nullptr
             ? rund::AccelCheck{false, "compute_scan_invalid"}
             : DescribeVulkanPipelineStatus(
                   scan->status, 1u, VulkanPipelineStatusRule::Exact, 0u,
                   mapping, source);
#else
  (void)resources;
  (void)source;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

[[nodiscard]] rund::AccelCheck DescribeVulkanScanPipelineTelemetry(
    const std::shared_ptr<void> &resources,
    VulkanPipelineTelemetrySource &source) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  source = {};
  const auto *const scan =
      static_cast<const VulkanKernelScanResources *>(resources.get());
  if (scan == nullptr) {
    return {false, "compute_plan_invalid"};
  }
  if (scan->control.iteration == 0u) {
    return {true, "ok"};
  }
  if (!scan->control.has_count() ||
      scan->logical_count.device_buffer == nullptr) {
    return {false, "compute_plan_invalid"};
  }
  source = VulkanPipelineTelemetrySource{
      .kind = VulkanPipelineTelemetryKind::ControlledCollective,
      .primary = scan->logical_count.device_buffer,
      .count = scan->logical_count.device_buffer,
      .control = scan->control,
      .count_offset = scan->logical_count.ref.offset_bytes +
                      scan->control.count_byte_offset,
      .capacity = scan->control.capacity,
  };
  return {true, "ok"};
#else
  (void)resources;
  (void)source;
  return {false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck FinishVulkanScan(VulkanAdapter &adapter,
                                  const std::shared_ptr<void> &resources) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const scan = static_cast<VulkanKernelScanResources *>(resources.get());
  if (scan == nullptr || scan->adapter != &adapter) {
    SetVulkanLastError(adapter, "compute_scan_invalid");
    return rund::AccelCheck{false, "compute_scan_invalid"};
  }
  const rund::kernel::u32 flags = VulkanScanStatusFlags(scan->status);
  if (flags != 0u) {
    const char *const reason = (flags & 2u) != 0u
                                   ? "compute_bounded_count_invalid"
                                   : "compute_scan_sum_overflow";
    SetVulkanLastError(adapter, reason);
    return rund::AccelCheck{false, reason};
  }
  const auto *const prepared = static_cast<const VulkanScanEncodeResources *>(
      scan->scan_resources.get());
  if (prepared == nullptr || prepared->dispatch_count == 0u) {
    SetVulkanLastError(adapter, "compute_scan_invalid");
    return rund::AccelCheck{false, "compute_scan_invalid"};
  }
  ::rund::detail::counter::Accumulate(adapter.dispatch_count,
                                      prepared->dispatch_count);
  SetVulkanLastError(adapter, "ok");
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
