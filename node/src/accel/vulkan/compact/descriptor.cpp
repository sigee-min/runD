#include "../descriptor/binding.hpp"
#include "local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
bool CreateVulkanCompactDescriptorSets(
    VulkanAdapter &adapter, VulkanCompactEncodeResources &resources,
    const VulkanResidentBufferResult &flags,
    const VulkanResidentBufferResult &output) {
  if (!AcquireVulkanCollectiveDescriptorSet(
          adapter, *resources.classify_pipeline, kCompactDescriptorCount,
          resources.classify_set) ||
      !AcquireVulkanCollectiveDescriptorSet(adapter, *resources.prefix_pipeline,
                                            kCompactDescriptorCount,
                                            resources.prefix_set) ||
      !AcquireVulkanCollectiveDescriptorSet(
          adapter, *resources.scatter_pipeline, kCompactDescriptorCount,
          resources.scatter_set)) {
    return false;
  }
  const std::array<VulkanStorageBinding, kCompactDescriptorCount> bindings{
      VulkanStorageBindingFor(resources.params),
      VulkanStorageBindingFor(flags.device_buffer, flags.ref),
      VulkanStorageBindingFor(resources.counts),
      VulkanStorageBindingFor(resources.offsets),
      VulkanStorageBindingFor(output.device_buffer, output.ref),
      VulkanStorageBindingFor(resources.status.device)};
  return WriteVulkanStorageDescriptorSet(adapter, resources.classify_set,
                                         bindings) &&
         WriteVulkanStorageDescriptorSet(adapter, resources.prefix_set,
                                         bindings) &&
         WriteVulkanStorageDescriptorSet(adapter, resources.scatter_set,
                                         bindings);
}
#endif

} // namespace rund::node::accel::detail
