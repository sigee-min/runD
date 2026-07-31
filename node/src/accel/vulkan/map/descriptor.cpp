#include "local.hpp"
#include "../descriptor.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
bool AllocateVulkanMapDescriptorSets(
    VulkanAdapter& adapter,
    const VulkanCachedPipeline& pipeline,
    const rund::kernel::u64 set_count,
    VkDescriptorPool& descriptor_pool,
    std::vector<VkDescriptorSet>& descriptor_sets) {
  return CreateVulkanStorageDescriptorSets(adapter, pipeline, set_count,
                                           descriptor_pool, descriptor_sets);
}
#endif

}  // namespace rund::node::accel::detail
