#include <accel/check.hpp>
#include <accel/device.hpp>

#include "local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

rund::AccelCheck ValidateVulkanKernelContext(const rund::AccelDevice &pick,
                                             VulkanKernelContext &out) {
  auto *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr) {
    return rund::AccelCheck{false, "accel_vulkan_unavailable"};
  }
  SetVulkanLastError(*adapter, "ok");
  out.adapter = adapter;
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
