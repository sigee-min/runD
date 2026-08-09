#include <accel/check.hpp>

#include "../../stencil/vulkan.hpp"
#include "../range/local.hpp"

#include <string_view>

namespace rund::node::accel::detail {

rund::AccelCheck FinishVulkanStencil(VulkanAdapter &adapter,
                                     const std::shared_ptr<void> &resources) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const rund::AccelCheck check = FinishVulkanRange(adapter, resources);
  if (!check.ok &&
      std::string_view{check.reason} == "compute_range_aggregate_invalid") {
    SetVulkanLastError(adapter, "compute_stencil_invalid");
    return rund::AccelCheck{false, "compute_stencil_invalid"};
  }
  return check;
#else
  (void)adapter;
  (void)resources;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
