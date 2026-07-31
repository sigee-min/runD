#include "../../../segmented/reduce/vulkan.hpp"

#include "../../collective/execute.hpp"

namespace rund::node::accel::detail {

rund::AccelCheck
ExecuteVulkanSegmentedReduce(const rund::AccelDevice &pick,
                             const rund::kernel::SegmentedReduceDesc &desc,
                             const rund::kernel::SegmentedReducePlan &plan,
                             const rund::kernel::ComputeDomain domain,
                             const SegmentedReduceBinds &bindings) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const auto prepare =
      [domain](const rund::AccelDevice &device,
               const rund::kernel::SegmentedReduceDesc &operation,
               const rund::kernel::SegmentedReducePlan &planned,
               const SegmentedReduceBinds &resident,
               std::shared_ptr<void> &resources) {
        return PrepareVulkanSegmentedReduce(device, operation, planned, domain,
                                            resident, resources);
      };
  return ExecuteVulkanCollective(pick, desc, plan, bindings, prepare,
                                 EncodeVulkanSegmentedReduce,
                                 FinishVulkanSegmentedReduce);
#else
  (void)domain;
  return RejectVulkanCollectiveExecute(pick, desc, plan, bindings);
#endif
}

} // namespace rund::node::accel::detail
