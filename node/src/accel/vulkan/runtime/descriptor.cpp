#include "local.hpp"
#include "../descriptor.hpp"
#include <rund/counter.hpp>

#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
VkDescriptorSet DescriptorSetForPipeline(VulkanAdapter& adapter,
                                         VulkanCachedPipeline& pipeline) {
  if (pipeline.descriptor_pool != VK_NULL_HANDLE &&
      pipeline.descriptor_set != VK_NULL_HANDLE) {
    ::rund::detail::counter::Accumulate(adapter.descriptor_reuse_hit_count, 1u);
    return pipeline.descriptor_set;
  }
  if (pipeline.descriptor_pool != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(adapter.device, pipeline.descriptor_pool, nullptr);
    pipeline.descriptor_pool = VK_NULL_HANDLE;
    pipeline.descriptor_set = VK_NULL_HANDLE;
  }
  const rund::kernel::u64 value_count =
      pipeline.input_buffer_count + pipeline.output_buffer_count;
  if (value_count < pipeline.input_buffer_count ||
      pipeline.output_buffer_count == 0u ||
      value_count > static_cast<rund::kernel::u64>(
                        std::numeric_limits<std::uint32_t>::max() - 1u)) {
    SetVulkanLastError(adapter, "accel_vulkan_descriptor_unavailable");
    return VK_NULL_HANDLE;
  }
  const std::uint32_t descriptor_count =
      static_cast<std::uint32_t>(value_count + 1u);
  if (!CreateVulkanStorageDescriptorSet(adapter, pipeline, descriptor_count,
                                        pipeline.descriptor_pool,
                                        pipeline.descriptor_set)) {
    if (pipeline.descriptor_pool != VK_NULL_HANDLE) {
      vkDestroyDescriptorPool(adapter.device, pipeline.descriptor_pool,
                              nullptr);
      pipeline.descriptor_pool = VK_NULL_HANDLE;
    }
    pipeline.descriptor_set = VK_NULL_HANDLE;
    return VK_NULL_HANDLE;
  }
  return pipeline.descriptor_set;
}
#endif

}  // namespace rund::node::accel::detail
