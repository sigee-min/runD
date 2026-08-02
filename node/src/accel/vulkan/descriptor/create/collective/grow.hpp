#pragma once

#include "capacity.hpp"
#include "reuse.hpp"
#include "../layout/sets.hpp"

#include <cstdint>
#include <limits>
namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool GrowCollectiveDescriptorSets(VulkanAdapter& adapter,
    VulkanCollectivePipeline& pipeline,
    const std::uint32_t descriptor_count,
    const std::uint64_t set_count) {
  if (set_count > static_cast<std::uint64_t>(
          std::numeric_limits<std::size_t>::max())) {
    SetVulkanLastError(adapter, "accel_vulkan_descriptor_unavailable");
    return false;
  }
  const std::size_t old = pipeline.descriptor_sets.size();
  const std::size_t target = static_cast<std::size_t>(set_count);
  if (target <= old) { return true; }
  if (!ReserveCollectiveDescriptorStorage(adapter, pipeline, target))
    return false;
  VkDescriptorPool pool = VK_NULL_HANDLE;
  pipeline.descriptor_sets.resize(target);
  pipeline.descriptor_leased.resize(target, 0u);
  if (!CreateDescriptorSetsWithLayout(adapter, descriptor_count, target - old,
                                      pipeline.descriptor_set_layout, pool,
                                      pipeline.descriptor_sets.data() + old)) {
    if (pool != VK_NULL_HANDLE)
      vkDestroyDescriptorPool(adapter.device, pool, nullptr);
    pipeline.descriptor_sets.resize(old);
    pipeline.descriptor_leased.resize(old);
    return false;
  }
  pipeline.descriptor_pools.push_back(pool);
  return true;
}

#endif

}  // namespace rund::node::accel::detail
