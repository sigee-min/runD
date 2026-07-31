#include <accel/check.hpp>

#include "../../segmented/status.hpp"
#include "../collective/finish.hpp"
#include "../kernel/ops/status.hpp"
#include "local.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelCheck
DescribeVulkanSegmentedPipelineStatus(const std::shared_ptr<void> &resources,
                                      VulkanPipelineStatusSource &source) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const scan =
      static_cast<VulkanSegmentedScanEncodeResources *>(resources.get());
  constexpr std::array mapping{
      VulkanPipelineStatusMapping{
          1u, rund::compute::Reason::SegmentedScanSumOverflow},
      VulkanPipelineStatusMapping{
          2u, rund::compute::Reason::SegmentedScanSegmentInvalid},
  };
  return scan == nullptr
             ? rund::AccelCheck{false, "compute_segmented_scan_invalid"}
             : DescribeVulkanPipelineStatus(scan->status, 1u,
                                            VulkanPipelineStatusRule::Exact, 0u,
                                            mapping, source);
#else
  (void)resources;
  (void)source;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck
FinishVulkanSegmentedScan(VulkanAdapter &adapter,
                          const std::shared_ptr<void> &resources) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanSegmentedScanEncodeResources *scan = nullptr;
  rund::AccelCheck check = LoadVulkanFinishResources(
      adapter, resources, "compute_segmented_scan_invalid", scan);
  if (!check.ok) {
    return check;
  }
  const rund::kernel::u32 *status = nullptr;
  check = ReadVulkanStatusU32(adapter, scan->status, status);
  if (!check.ok) {
    return check;
  }
  check = SegmentedScanStatus(*status);
  if (!check.ok) {
    SetVulkanLastError(adapter, check.reason);
    return check;
  }
  return AcceptVulkanDispatches(adapter, scan->dispatch_count);
#else
  (void)adapter;
  (void)resources;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
