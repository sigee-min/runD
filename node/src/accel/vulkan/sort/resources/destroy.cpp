#include "../local/api.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
void DestroyVulkanSortEncodeResources(void *const raw) {
  auto *const resources = static_cast<VulkanSortEncodeResources *>(raw);
  if (resources == nullptr) {
    return;
  }
  VulkanAdapter *const adapter = resources->adapter;
  if (adapter != nullptr) {
    ReleaseVulkanBuffer(*adapter, resources->params);
    ReleaseVulkanBuffer(*adapter, resources->temp_keys);
    ReleaseVulkanBuffer(*adapter, resources->temp_values);
    ReleaseVulkanBuffer(*adapter, resources->block_counts);
    ReleaseVulkanBuffer(*adapter, resources->block_offsets);
    ReleaseVulkanBuffer(*adapter, resources->dispatch_args);
    ReleaseVulkanStatus(*adapter, resources->status);
  }
  delete resources;
}
#endif

} // namespace rund::node::accel::detail
