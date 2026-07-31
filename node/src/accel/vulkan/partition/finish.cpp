#include <accel/check.hpp>

#include <rund/counter.hpp>
#include "../kernel/ops/status.hpp"
#include "../../scan/count.hpp"
#include "local.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelCheck DescribeVulkanPartitionPipelineStatus(
    const std::shared_ptr<void> &resources,
    VulkanPipelineStatusSource &source) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const partition =
      static_cast<VulkanPartitionEncodeResources *>(resources.get());
  constexpr std::array mapping{VulkanPipelineStatusMapping{
      std::numeric_limits<std::uint32_t>::max(),
      rund::compute::Reason::ScanSumOverflow}};
  return partition == nullptr
             ? rund::AccelCheck{false, "compute_partition_invalid"}
             : DescribeVulkanPipelineStatus(
                   partition->false_status, 1u,
                   VulkanPipelineStatusRule::BitFlags, 0u, mapping, source);
#else
  (void)resources;
  (void)source;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck FinishVulkanPartition(VulkanAdapter &adapter,
                                       const std::shared_ptr<void> &resources) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const partition =
      static_cast<VulkanPartitionEncodeResources *>(resources.get());
  if (partition == nullptr || partition->adapter != &adapter) {
    SetVulkanLastError(adapter, "compute_partition_invalid");
    return rund::AccelCheck{false, "compute_partition_invalid"};
  }
  if (!VulkanScanStatusOk(partition->false_status)) {
    SetVulkanLastError(adapter, "compute_scan_sum_overflow");
    return rund::AccelCheck{false, "compute_scan_sum_overflow"};
  }
  ::rund::detail::counter::Accumulate(
      adapter.dispatch_count,
      2u + EncodedScanDispatchCount(partition->scan_plan));
  SetVulkanLastError(adapter, "ok");
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
