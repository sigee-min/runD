#pragma once

#include <accel/check.hpp>

#include "local.hpp"

namespace rund::node::accel::detail {

rund::AccelCheck EncodeVulkanScan(VulkanAdapter &adapter,
                                  const std::shared_ptr<void> &resources,
                                  void *command_buffer) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const scan = static_cast<VulkanKernelScanResources *>(resources.get());
  if (scan == nullptr || scan->adapter != &adapter ||
      scan->scan_resources == nullptr) {
    SetVulkanLastError(adapter, "compute_scan_invalid");
    return rund::AccelCheck{false, "compute_scan_invalid"};
  }
  return EncodeVulkanScanBuffers(adapter, scan->scan_resources, command_buffer);
#else
  (void)adapter;
  (void)resources;
  (void)command_buffer;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
