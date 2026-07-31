#pragma once

#include "layout.hpp"
#include "../../descriptor.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

bool CreateVulkanStorageDescriptorSets(
    VulkanAdapter& adapter,
    const VulkanCachedPipeline& pipeline,
    const std::uint64_t set_count,
    VkDescriptorPool& pool,
    std::vector<VkDescriptorSet>& sets) {
  if (set_count == 0u ||
      set_count > static_cast<std::uint64_t>(
                      std::numeric_limits<std::size_t>::max()) ||
      set_count > std::numeric_limits<std::uint32_t>::max() ||
      pipeline.output_buffer_count == 0u ||
      pipeline.output_buffer_count >
          static_cast<rund::kernel::u64>(
              std::numeric_limits<std::uint32_t>::max() - 1u) ||
      pipeline.input_buffer_count >
          static_cast<rund::kernel::u64>(
              std::numeric_limits<std::uint32_t>::max() - 1u) -
              pipeline.output_buffer_count) {
    SetVulkanLastError(adapter, "accel_vulkan_descriptor_unavailable");
    return false;
  }

  const std::uint32_t descriptor_count =
      static_cast<std::uint32_t>(pipeline.input_buffer_count +
                                 pipeline.output_buffer_count + 1u);
  sets.resize(static_cast<std::size_t>(set_count));
  if (CreateDescriptorSetsWithLayout(adapter, descriptor_count, set_count,
                                     pipeline.descriptor_set_layout, pool,
                                     sets.data())) {
    return true;
  }

  if (pool != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(adapter.device, pool, nullptr);
    pool = VK_NULL_HANDLE;
  }
  sets.clear();
  return false;
}

#endif

}  // namespace rund::node::accel::detail
