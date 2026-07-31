#pragma once

#include "../../../../clock.hpp"
#include "../../../descriptor.hpp"
#include "../../../runtime/counter.hpp"
#include "layouts.hpp"
#include "pool.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool CreateDescriptorSetsWithLayout(
    VulkanAdapter &adapter, const std::uint32_t descriptor_count,
    const std::uint64_t set_count, const VkDescriptorSetLayout layout,
    VkDescriptorPool &pool, VkDescriptorSet *const sets) {
  if (set_count == 0u || sets == nullptr || layout == VK_NULL_HANDLE ||
      set_count > std::numeric_limits<std::uint32_t>::max()) {
    SetVulkanLastError(adapter, "accel_vulkan_descriptor_unavailable");
    return false;
  }

  const std::uint64_t begin = MonotonicNanoseconds();
  const auto set_count32 = static_cast<std::uint32_t>(set_count);
  if (!CreateDescriptorPoolForSets(adapter, descriptor_count, set_count32,
                                   pool)) {
    return false;
  }

  SetLayoutScratch layouts{};
  FillSetLayouts(layouts, layout, set_count32);
  VkDescriptorSetAllocateInfo alloc{};
  alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  alloc.descriptorPool = pool;
  alloc.descriptorSetCount = set_count32;
  alloc.pSetLayouts = layouts.data;
  if (vkAllocateDescriptorSets(adapter.device, &alloc, sets) != VK_SUCCESS) {
    SetVulkanLastError(adapter, "accel_vulkan_descriptor_unavailable");
    return false;
  }

  ::rund::detail::counter::Accumulate(adapter.descriptor_set_allocate_count,
                                      set_count);
  RecordVulkanDescriptorSetupNs(adapter, MonotonicNanoseconds() - begin);
  return true;
}

#endif

} // namespace rund::node::accel::detail
