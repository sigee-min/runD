#include <accel/check.hpp>

#include <rund/counter.hpp>
#include "../kernel/ops/status.hpp"
#include "local.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelCheck DescribeVulkanCompactPipelineStatus(
    const std::shared_ptr<void> &resources,
    VulkanPipelineStatusSource &source) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const compact =
      static_cast<VulkanCompactEncodeResources *>(resources.get());
  constexpr std::array mapping{VulkanPipelineStatusMapping{
      std::numeric_limits<std::uint32_t>::max(),
      rund::compute::Reason::CompactCapacityInsufficient}};
  return compact == nullptr
             ? rund::AccelCheck{false, "compute_plan_invalid"}
             : DescribeVulkanPipelineStatus(
                   compact->status, 1u, VulkanPipelineStatusRule::BitFlags, 0u,
                   mapping, source);
#else
  (void)resources;
  (void)source;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck FinishVulkanCompact(VulkanAdapter &adapter,
                                     const std::shared_ptr<void> &resources) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const compact =
      static_cast<VulkanCompactEncodeResources *>(resources.get());
  if (compact == nullptr || compact->adapter != &adapter ||
      compact->block_count == 0u) {
    SetVulkanLastError(adapter, "compute_plan_invalid");
    return rund::AccelCheck{false, "compute_plan_invalid"};
  }

  if (compact->plan.status_bytes != 0u) {
    const auto *const status = VulkanStatusValue(compact->status);
    if (status == nullptr) {
      SetVulkanLastError(adapter, "accel_vulkan_memory_unavailable");
      return rund::AccelCheck{false, "accel_vulkan_memory_unavailable"};
    }
    if (*status != 0u) {
      SetVulkanLastError(adapter, "compute_compact_capacity_insufficient");
      return rund::AccelCheck{false,
                              "compute_compact_capacity_insufficient"};
    }
  }
  ::rund::detail::counter::Accumulate(adapter.dispatch_count,
                                      kCompactDispatchCount);
  SetVulkanLastError(adapter, "ok");
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
