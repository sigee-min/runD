#include "../descriptor/binding.hpp"
#include "local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
bool CreateVulkanPartitionDescriptorSets(
    VulkanAdapter &adapter, VulkanPartitionEncodeResources &resources) {
  if (!AcquireVulkanCollectiveDescriptorSet(
          adapter, *resources.classify_pipeline,
          kPartitionClassifyDescriptorCount, resources.classify_set) ||
      !AcquireVulkanCollectiveDescriptorSet(
          adapter, *resources.scatter_pipeline,
          kPartitionScatterDescriptorCount, resources.scatter_set)) {
    return false;
  }
  const bool classify = WriteVulkanStorageDescriptorSet(
      adapter, resources.classify_set,
      std::array<VulkanStorageBinding, kPartitionClassifyDescriptorCount>{
          VulkanStorageBindingFor(resources.params),
          VulkanStorageBindingFor(resources.flags, resources.flags_ref),
          VulkanStorageBindingFor(resources.false_bits)});
  const bool scatter = WriteVulkanStorageDescriptorSet(
      adapter, resources.scatter_set,
      std::array<VulkanStorageBinding, kPartitionScatterDescriptorCount>{
          VulkanStorageBindingFor(resources.params),
          VulkanStorageBindingFor(resources.flags, resources.flags_ref),
          VulkanStorageBindingFor(resources.values, resources.values_ref),
          VulkanStorageBindingFor(resources.output, resources.output_ref),
          VulkanStorageBindingFor(resources.false_offsets)});
  return classify && scatter;
}
#endif

} // namespace rund::node::accel::detail
