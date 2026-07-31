#include <accel/check.hpp>

#include "../collective/finish.hpp"
#include "local.hpp"

namespace rund::node::accel::detail {

rund::AccelCheck FinishVulkanStencil(VulkanAdapter &adapter,
                                     const std::shared_ptr<void> &resources) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanStencilEncodeResources *stencil = nullptr;
  const rund::AccelCheck check = LoadVulkanFinishResources(
      adapter, resources, "compute_stencil_invalid", stencil);
  if (!check.ok) {
    return check;
  }
  return AcceptVulkanDispatches(adapter, stencil->plan.pass_count);
#else
  (void)adapter;
  (void)resources;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
