#include "local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
bool CreateVulkanScanDescriptorSets(VulkanAdapter &adapter,
                                    VulkanScanEncodeResources &resources,
                                    const VulkanStorageBinding input,
                                    const VulkanStorageBinding logical_count) {
  if (!AcquireVulkanCollectiveDescriptorSet(adapter, *resources.block,
                                            kScanDescriptorCount,
                                            resources.block_set) ||
      (resources.pass_count == 2u &&
       (!AcquireVulkanCollectiveDescriptorSet(adapter, *resources.prefix,
                                              kScanDescriptorCount,
                                              resources.prefix_set) ||
        !AcquireVulkanCollectiveDescriptorSet(adapter, *resources.offset,
                                              kScanDescriptorCount,
                                              resources.offset_set)))) {
    return false;
  }
  bool ready = WriteVulkanStorageDescriptorSet(
      adapter, resources.block_set,
      std::array<VulkanStorageBinding, kScanDescriptorCount>{
          VulkanStorageBindingFor(resources.params), input,
          resources.output_binding, VulkanStorageBindingFor(resources.totals),
          VulkanStorageBindingFor(resources.status->device), logical_count});
  if (resources.pass_count == 2u) {
    ready = ready && WriteVulkanStorageDescriptorSet(
                         adapter, resources.prefix_set,
                         std::array<VulkanStorageBinding, kScanDescriptorCount>{
                             VulkanStorageBindingFor(resources.params), input,
                             resources.output_binding,
                             VulkanStorageBindingFor(resources.totals),
                             VulkanStorageBindingFor(resources.status->device),
                             logical_count});
    ready = ready && WriteVulkanStorageDescriptorSet(
                         adapter, resources.offset_set,
                         std::array<VulkanStorageBinding, kScanDescriptorCount>{
                             VulkanStorageBindingFor(resources.params), input,
                             resources.output_binding,
                             VulkanStorageBindingFor(resources.totals),
                             VulkanStorageBindingFor(resources.status->device),
                             logical_count});
  }
  return ready;
}
#endif

} // namespace rund::node::accel::detail
