#include "local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
bool CreateVulkanScatterDescriptorSet(VulkanAdapter &adapter,
                                      VulkanScatterEncodeResources &resources) {
  if (!AcquireVulkanCollectiveDescriptorSet(adapter, *resources.pipeline,
                                            kScatterDescriptorCount,
                                            resources.descriptor_set)) {
    return false;
  }
  return WriteVulkanStorageDescriptorSet(
      adapter, resources.descriptor_set,
      std::array<VulkanStorageBinding, kScatterDescriptorCount>{
          VulkanStorageBindingFor(resources.params), resources.values_binding,
          resources.indices_binding, resources.output_binding,
          VulkanStorageBindingFor(resources.status.device)});
}
#endif

} // namespace rund::node::accel::detail
