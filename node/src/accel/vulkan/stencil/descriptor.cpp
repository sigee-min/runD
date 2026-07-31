#include "local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
bool CreateVulkanStencilDescriptorSet(VulkanAdapter &adapter,
                                      VulkanStencilEncodeResources &resources) {
  if (!AcquireVulkanCollectiveDescriptorSet(adapter, *resources.pipeline,
                                            kStencilDescriptorCount,
                                            resources.descriptor_set)) {
    return false;
  }
  return WriteVulkanStorageDescriptorSet(
      adapter, resources.descriptor_set,
      std::array<VulkanStorageBinding, kStencilDescriptorCount>{
          VulkanStorageBindingFor(resources.params), resources.input_binding,
          resources.output_binding});
}
#endif

} // namespace rund::node::accel::detail
