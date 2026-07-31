#include "local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] bool CreateSegmentedSet(VulkanAdapter &adapter,
                                      VulkanCollectivePipeline *pipeline,
                                      VkDescriptorSet &set,
                                      VulkanSegmentedScanEncodeResources &r) {
  if (!AcquireVulkanCollectiveDescriptorSet(
          adapter, *pipeline, kSegmentedScanDescriptorCount, set)) {
    return false;
  }
  return WriteVulkanStorageDescriptorSet(
      adapter, set,
      std::array<VulkanStorageBinding, kSegmentedScanDescriptorCount>{
          VulkanStorageBindingFor(r.params), r.input_binding, r.heads_binding,
          r.output_binding, VulkanStorageBindingFor(r.offsets),
          VulkanStorageBindingFor(r.first_heads),
          VulkanStorageBindingFor(r.status.device)});
}

} // namespace

bool CreateVulkanSegmentedScanDescriptorSet(
    VulkanAdapter &adapter, VulkanSegmentedScanEncodeResources &resources) {
  if (!CreateSegmentedSet(adapter, resources.block, resources.block_set,
                          resources)) {
    return false;
  }
  if (resources.plan.pass_count == 1u) {
    return true;
  }
  return CreateSegmentedSet(adapter, resources.prefix, resources.prefix_set,
                            resources) &&
         CreateSegmentedSet(adapter, resources.offset, resources.offset_set,
                            resources);
}
#endif

} // namespace rund::node::accel::detail
