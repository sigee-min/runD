#include <accel/check.hpp>
#include <accel/device.hpp>

#include "local.hpp"

#include "../collective/execute.hpp"

namespace rund::node::accel::detail {

rund::AccelCheck
ExecuteVulkanHistogram(const rund::AccelDevice &pick,
                       const rund::kernel::HistogramDesc &desc,
                       const rund::kernel::HistogramPlan &plan,
                       const HistogramBinds &bindings) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  return ExecuteVulkanCollective(pick, desc, plan, bindings,
                                 [](const rund::AccelDevice &device,
                                    const rund::kernel::HistogramDesc &operation,
                                    const rund::kernel::HistogramPlan &prepared,
                                    const HistogramBinds &resident,
                                    std::shared_ptr<void> &resources) {
                                   return PrepareVulkanHistogram(
                                       device, operation, prepared, resident,
                                       resources, nullptr);
                                 },
                                 EncodeVulkanHistogram,
                                 FinishVulkanHistogram);
#else
  return RejectVulkanCollectiveExecute(pick, desc, plan, bindings);
#endif
}

} // namespace rund::node::accel::detail
