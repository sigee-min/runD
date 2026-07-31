#include <accel/check.hpp>

#include "../collective/finish.hpp"
#include "../kernel/ops/status.hpp"
#include "local.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelCheck DescribeVulkanScatterPipelineStatus(
    const std::shared_ptr<void> &resources,
    VulkanPipelineStatusSource &source) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const scatter =
      static_cast<VulkanScatterEncodeResources *>(resources.get());
  constexpr std::array mapping{
      VulkanPipelineStatusMapping{
          kScatterReasonOutOfRange,
          rund::compute::Reason::ScatterIndexOutOfRange},
      VulkanPipelineStatusMapping{
          kScatterReasonDuplicate,
          rund::compute::Reason::ScatterDuplicateIndex},
  };
  return scatter == nullptr
             ? rund::AccelCheck{false, "compute_scatter_invalid"}
             : DescribeVulkanPipelineStatus(
                   scatter->status, 1u, VulkanPipelineStatusRule::LowBit,
                   kScatterStatusOk, mapping, source);
#else
  (void)resources;
  (void)source;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck FinishVulkanScatter(VulkanAdapter &adapter,
                                     const std::shared_ptr<void> &resources) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanScatterEncodeResources *scatter = nullptr;
  rund::AccelCheck check = LoadVulkanFinishResources(
      adapter, resources, "compute_scatter_invalid", scatter);
  if (!check.ok) {
    return check;
  }
  const rund::kernel::u32 *status = nullptr;
  check = ReadVulkanStatusU32(adapter, scatter->status, status);
  if (!check.ok) {
    return check;
  }
  if (*status != kScatterStatusOk) {
    const rund::kernel::u32 reason = *status & 1u;
    const char *const failure = reason == kScatterReasonOutOfRange
                                    ? "compute_scatter_index_out_of_range"
                                    : (reason == kScatterReasonDuplicate
                                           ? "compute_scatter_duplicate_index"
                                           : "compute_scatter_invalid");
    SetVulkanLastError(adapter, failure);
    return rund::AccelCheck{false, failure};
  }
  return AcceptVulkanDispatches(adapter, scatter->plan.pass_count);
#else
  (void)adapter;
  (void)resources;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
