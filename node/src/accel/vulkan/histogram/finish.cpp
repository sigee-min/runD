#include <accel/check.hpp>

#include "local.hpp"

#include "../collective/finish.hpp"
#include "../kernel/ops/status.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelCheck DescribeVulkanHistogramPipelineStatus(
    const std::shared_ptr<void> &resources,
    VulkanPipelineStatusSource &source) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const histogram =
      static_cast<VulkanHistogramEncodeResources *>(resources.get());
  constexpr std::array mapping{VulkanPipelineStatusMapping{
      kHistogramReasonBinInvalid, rund::compute::Reason::HistogramBinInvalid}};
  return histogram == nullptr
             ? rund::AccelCheck{false, "compute_histogram_invalid"}
             : DescribeVulkanPipelineStatus(
                   histogram->status, 1u,
                   VulkanPipelineStatusRule::AnyFailure,
                   kHistogramStatusOk, mapping, source);
#else
  (void)resources;
  (void)source;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck FinishVulkanHistogram(VulkanAdapter &adapter,
                                       const std::shared_ptr<void> &resources) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanHistogramEncodeResources *histogram = nullptr;
  rund::AccelCheck check = LoadVulkanFinishResources(
      adapter, resources, "compute_histogram_invalid", histogram);
  if (!check.ok) {
    return check;
  }
  const rund::kernel::u32 *status = nullptr;
  check = ReadVulkanStatusU32(adapter, histogram->status, status);
  if (!check.ok) {
    return check;
  }
  if (*status != kHistogramStatusOk) {
    SetVulkanLastError(adapter, "compute_histogram_bin_invalid");
    return rund::AccelCheck{false, "compute_histogram_bin_invalid"};
  }
  return AcceptVulkanDispatches(adapter, histogram->plan.pass_count);
#else
  (void)adapter;
  (void)resources;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
