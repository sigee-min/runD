#pragma once

#include <accel/check.hpp>

#include "../adapter/api.hpp"
#include "../status.hpp"
#include <rund/counter.hpp>

#include <memory>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
template <class Resources>
rund::AccelCheck
LoadVulkanFinishResources(VulkanAdapter &adapter,
                          const std::shared_ptr<void> &resources,
                          const char *invalid_reason, Resources *&out) {
  out = static_cast<Resources *>(resources.get());
  if (out == nullptr || out->adapter != &adapter) {
    SetVulkanLastError(adapter, invalid_reason);
    return rund::AccelCheck{false, invalid_reason};
  }
  return rund::AccelCheck{true, "ok"};
}

inline rund::AccelCheck ReadVulkanStatusU32(VulkanAdapter &adapter,
                                            const VulkanStatus &status_buffer,
                                            const rund::kernel::u32 *&status) {
  status = VulkanStatusValue(status_buffer);
  if (status == nullptr) {
    SetVulkanLastError(adapter, "accel_vulkan_memory_unavailable");
    return rund::AccelCheck{false, "accel_vulkan_memory_unavailable"};
  }
  return rund::AccelCheck{true, "ok"};
}

inline rund::AccelCheck
AcceptVulkanDispatches(VulkanAdapter &adapter,
                       const rund::kernel::u64 dispatches) {
  ::rund::detail::counter::Accumulate(adapter.dispatch_count, dispatches);
  SetVulkanLastError(adapter, "ok");
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
