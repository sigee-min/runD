#include "local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
bool CreateVulkanRangeDescriptors(VulkanAdapter &adapter,
                                  VulkanRangeResources &resources,
                                  const std::uint32_t stage_index) {
  if (stage_index >= resources.stage_count ||
      resources.pipelines[stage_index] == nullptr) {
    return false;
  }
  const std::uint32_t descriptor_count = RangeDescriptorCount(resources.range);
  if (!AcquireVulkanCollectiveDescriptorSet(
          adapter, *resources.pipelines[stage_index], descriptor_count,
          resources.descriptor_sets[stage_index])) {
    return false;
  }
  const VulkanStorageBinding params =
      VulkanStorageBindingFor(resources.params[stage_index]);
  if (descriptor_count == 3u) {
    return WriteVulkanStorageDescriptorSet(
        adapter, resources.descriptor_sets[stage_index],
        std::array<VulkanStorageBinding, 3u>{params, resources.input_binding,
                                             resources.output_binding});
  }
  const VulkanBuffer *scratch0 = nullptr;
  const VulkanBuffer *scratch1 = nullptr;
  return descriptor_count == 5u &&
         VulkanRangeScratch(resources, stage_index, scratch0, scratch1) &&
         scratch0 != nullptr && scratch1 != nullptr &&
         WriteVulkanStorageDescriptorSet(
             adapter, resources.descriptor_sets[stage_index],
             std::array<VulkanStorageBinding, 5u>{
                 params, resources.input_binding, resources.output_binding,
                 VulkanStorageBindingFor(*scratch0),
                 VulkanStorageBindingFor(*scratch1)});
}
#endif

} // namespace rund::node::accel::detail
