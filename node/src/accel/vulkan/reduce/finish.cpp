#include <accel/check.hpp>

#include "../../reduce/vulkan.hpp"
#include "../collective/finish.hpp"
#include "../kernel/ops/status.hpp"
#include "local.hpp"
namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelCheck DescribeVulkanReducePipelineStatus(
    const std::shared_ptr<void> &resources,
    VulkanPipelineStatusSource &source) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const reduce =
      static_cast<VulkanReduceEncodeResources *>(resources.get());
  if (reduce == nullptr) {
    return rund::AccelCheck{false, "compute_reduce_invalid"};
  }
  const auto arithmetic =
      reduce->plan.op == rund::kernel::ReduceOp::CountNonzero
          ? rund::compute::Reason::ReduceCountOverflow
          : rund::compute::Reason::ReduceSumOverflow;
  const std::array mapping{
      VulkanPipelineStatusMapping{1u, arithmetic},
      VulkanPipelineStatusMapping{2u,
                                  rund::compute::Reason::BoundedCountInvalid},
      VulkanPipelineStatusMapping{3u,
                                  rund::compute::Reason::BoundedCountInvalid},
      VulkanPipelineStatusMapping{4u,
                                  rund::compute::Reason::ReduceCountZero},
      VulkanPipelineStatusMapping{5u,
                                  rund::compute::Reason::ReduceCountZero},
      VulkanPipelineStatusMapping{6u,
                                  rund::compute::Reason::BoundedCountInvalid},
      VulkanPipelineStatusMapping{7u,
                                  rund::compute::Reason::BoundedCountInvalid},
  };
  return DescribeVulkanPipelineStatus(
      reduce->status, 1u, VulkanPipelineStatusRule::Exact, 0u, mapping,
      source);
#else
  (void)resources;
  (void)source;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck FinishVulkanReduce(VulkanAdapter &adapter,
                                    const std::shared_ptr<void> &resources) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanReduceEncodeResources *reduce = nullptr;
  rund::AccelCheck check = LoadVulkanFinishResources(
      adapter, resources, "compute_reduce_invalid", reduce);
  if (!check.ok) {
    return check;
  }
  const rund::kernel::u32 *status = nullptr;
  check = ReadVulkanStatusU32(adapter, reduce->status, status);
  if (!check.ok) {
    return check;
  }
  if (*status != 0u) {
    const char *const reason =
        (*status & 2u) != 0u
            ? "compute_bounded_count_invalid"
            : ((*status & 4u) != 0u
                   ? "compute_reduce_count_zero"
                   : (reduce->plan.op == rund::kernel::ReduceOp::CountNonzero
                          ? "compute_reduce_count_overflow"
                          : "compute_reduce_sum_overflow"));
    SetVulkanLastError(adapter, reason);
    return rund::AccelCheck{false, reason};
  }
  return AcceptVulkanDispatches(adapter, reduce->plan.pass_count);
#else
  (void)adapter;
  (void)resources;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}
} // namespace rund::node::accel::detail
