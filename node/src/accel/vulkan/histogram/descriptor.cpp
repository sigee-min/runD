#include "local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
bool CreateVulkanHistogramDescriptorSets(
    VulkanAdapter &adapter, VulkanHistogramEncodeResources &resources) {
  if (!AcquireVulkanCollectiveDescriptorSet(adapter, *resources.clear_pipeline,
                                            kHistogramDescriptorCount,
                                            resources.clear_descriptor_set) ||
      !AcquireVulkanCollectiveDescriptorSet(adapter, *resources.count_pipeline,
                                            kHistogramDescriptorCount,
                                            resources.count_descriptor_set)) {
    return false;
  }
  const std::array<VulkanStorageBinding, kHistogramDescriptorCount> buffers{
      VulkanStorageBindingFor(resources.params), resources.bins_binding,
      resources.counts_binding,
      VulkanStorageBindingFor(resources.status.device)};
  return WriteVulkanStorageDescriptorSet(
             adapter, resources.clear_descriptor_set, buffers) &&
         WriteVulkanStorageDescriptorSet(
             adapter, resources.count_descriptor_set, buffers);
}
#endif

} // namespace rund::node::accel::detail
