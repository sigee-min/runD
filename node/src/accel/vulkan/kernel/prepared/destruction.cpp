#include "local.hpp"

#include "../trace.hpp"

#include <mutex>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

void DestroyPreparedVulkanKernelResources(void *const raw) {
  auto *const resources = static_cast<VulkanKernelResources *>(raw);
  if (resources == nullptr) {
    return;
  }
  VulkanAdapter *const adapter = resources->adapter;
  if (adapter != nullptr) {
    std::lock_guard<std::mutex> lock{adapter->mutex};
    DestroyVulkanKernelCommand(*adapter, *resources);
    DestroyVulkanDispatchTrace(*adapter, resources->trace);
    resources->release();
    for (const VulkanCollectiveDescriptorLease &lease :
         resources->descriptor_leases) {
      if (lease.pipeline != nullptr &&
          lease.slot < lease.pipeline->descriptor_leased.size()) {
        lease.pipeline->descriptor_leased[lease.slot] = false;
      }
    }
  }
  delete resources;
}

#endif

} // namespace rund::node::accel::detail
