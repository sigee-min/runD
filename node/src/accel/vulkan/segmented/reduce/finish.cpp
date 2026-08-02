#include "model.hpp"

#include "../../../segmented/reduce/status.hpp"
#include "../../../segmented/reduce/vulkan.hpp"

#include "../../collective/finish.hpp"
#include "../../kernel/ops/status.hpp"

#include <rund/compute/reason.hpp>

#include <array>

namespace rund::node::accel::detail {

rund::AccelCheck
FinishVulkanSegmentedReduce(VulkanAdapter &adapter,
                            const std::shared_ptr<void> &resources) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const state =
      static_cast<VulkanSegmentedReduceResources *>(resources.get());
  const rund::kernel::u32 *const status =
      state == nullptr ? nullptr : VulkanStatusValue(state->status);
  if (state == nullptr || state->adapter != &adapter || status == nullptr) {
    return {false, "compute_segmented_reduce_invalid"};
  }
  const rund::AccelCheck check = SegmentedReduceStatus(*status);
  return check.ok ? AcceptVulkanDispatches(adapter, 4u) : check;
#else
  (void)adapter;
  (void)resources;
  return {false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck DescribeVulkanSegmentedReducePipelineStatus(
    const std::shared_ptr<void> &resources,
    VulkanPipelineStatusSource &source) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const state =
      static_cast<VulkanSegmentedReduceResources *>(resources.get());
  constexpr auto segment = rund::compute::Reason::SegmentedReduceSegmentInvalid;
  constexpr auto sum = rund::compute::Reason::SegmentedReduceSumOverflow;
  constexpr auto count = rund::compute::Reason::SegmentedReduceCountOverflow;
  constexpr std::array mapping{
      VulkanPipelineStatusMapping{kSegmentInvalid, segment},
      VulkanPipelineStatusMapping{kSegmentSumOverflow, sum},
      VulkanPipelineStatusMapping{kSegmentInvalid | kSegmentSumOverflow,
                                  segment},
      VulkanPipelineStatusMapping{kSegmentCountOverflow, count},
      VulkanPipelineStatusMapping{kSegmentInvalid | kSegmentCountOverflow,
                                  segment},
      VulkanPipelineStatusMapping{kSegmentSumOverflow | kSegmentCountOverflow,
                                  sum},
      VulkanPipelineStatusMapping{kSegmentInvalid | kSegmentSumOverflow |
                                      kSegmentCountOverflow,
                                  segment},
  };
  return state == nullptr
             ? rund::AccelCheck{false, "compute_segmented_reduce_invalid"}
             : DescribeVulkanPipelineStatus(state->status, 1u,
                                            VulkanPipelineStatusRule::Exact, 0u,
                                            mapping, source);
#else
  (void)resources;
  (void)source;
  return {false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck DescribeVulkanSegmentedReduceCaptureDemand(
    const std::shared_ptr<void> &resources,
    std::uint64_t &indirect_dispatch_count) noexcept {
  indirect_dispatch_count = 0u;
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  if (static_cast<const VulkanSegmentedReduceResources *>(resources.get()) ==
      nullptr) {
    return {false, "compute_segmented_reduce_invalid"};
  }
  indirect_dispatch_count = 1u;
  return {true, "ok"};
#else
  (void)resources;
  return {false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
