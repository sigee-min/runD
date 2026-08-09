#include <accel/check.hpp>

#include "../collective/finish.hpp"
#include "local.hpp"

#include <string_view>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
rund::AccelCheck FinishVulkanRange(VulkanAdapter &adapter,
                                   const std::shared_ptr<void> &resources) {
  VulkanRangeResources *range = nullptr;
  const rund::AccelCheck check = LoadVulkanFinishResources(
      adapter, resources, "compute_range_aggregate_invalid", range);
  if (!check.ok) {
    return check;
  }
  return AcceptVulkanDispatches(adapter, range->range.stage_count());
}
#endif

} // namespace rund::node::accel::detail
