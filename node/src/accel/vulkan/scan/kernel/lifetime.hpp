#pragma once

#include "local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
void DestroyVulkanKernelScanResources(void* const raw) {
  auto* const resources = static_cast<VulkanKernelScanResources*>(raw);
  if (resources == nullptr) { return; }
  if (resources->adapter != nullptr &&
      resources->adapter->device != VK_NULL_HANDLE) {
    VulkanAdapter& adapter = *resources->adapter;
    resources->scan_resources.reset();
    ReleaseVulkanBuffer(adapter, resources->totals);
    ReleaseVulkanStatus(adapter, resources->status);
  }
  delete resources;
}
#endif

}  // namespace rund::node::accel::detail
