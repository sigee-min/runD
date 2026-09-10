#include "../local.hpp"

#include "../../status.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

void DestroyVulkanMapEncodeResources(void *const raw) {
  auto *const resources = static_cast<VulkanMapEncodeResources *>(raw);
  if (resources == nullptr) {
    return;
  }
  if (resources->adapter != nullptr) {
    ReleaseVulkanStatus(*resources->adapter, resources->control_status);
  }
  delete resources;
}

#endif

} // namespace rund::node::accel::detail
