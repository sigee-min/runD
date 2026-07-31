#include "local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
bool CreateVulkanReducePassDescriptorSet(
    VulkanAdapter &adapter, VulkanReduceEncodeResources &resources,
    const std::size_t index, const VulkanStorageBinding read_buffer) {
  if (!AcquireVulkanCollectiveDescriptorSet(adapter, *resources.pipeline,
                                            kReduceDescriptorCount,
                                            resources.descriptor_sets[index])) {
    return false;
  }
  const VkDeviceSize params_offset =
      static_cast<VkDeviceSize>(index) * resources.params_stride;
  return WriteVulkanStorageDescriptorSet(
      adapter, resources.descriptor_sets[index],
      std::array<VulkanStorageBinding, kReduceDescriptorCount>{
          VulkanStorageBinding{&resources.params, params_offset,
                               sizeof(ReducePassParams)},
          read_buffer, VulkanStorageBindingFor(resources.partial),
          resources.output, VulkanStorageBindingFor(resources.status.device),
          resources.logical_count});
}
#endif

} // namespace rund::node::accel::detail
