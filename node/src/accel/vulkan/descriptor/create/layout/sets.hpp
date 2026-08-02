#pragma once

#include "../../../../clock.hpp"
#include "../../../descriptor.hpp"
#include "../../../runtime/counter.hpp"
#include "pool.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

// Keep cold descriptor allocation stack-bounded while amortizing Vulkan calls.
// 256 handles occupy 2 KiB on a 64-bit ABI; unlike the removed heap scratch,
// this bound is independent of route count and cannot fail after pool creation.
inline constexpr std::uint32_t kDescriptorSetAllocationBatch = 256u;

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

  std::array<VkDescriptorSetLayout, kDescriptorSetAllocationBatch> layouts{};
  layouts.fill(layout);
  std::uint32_t allocated = 0u;
  while (allocated != set_count32) {
    const std::uint32_t batch =
        std::min(kDescriptorSetAllocationBatch, set_count32 - allocated);
    VkDescriptorSetAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc.descriptorPool = pool;
    alloc.descriptorSetCount = batch;
    alloc.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(adapter.device, &alloc, sets + allocated) !=
        VK_SUCCESS) {
      vkDestroyDescriptorPool(adapter.device, pool, nullptr);
      pool = VK_NULL_HANDLE;
      std::fill_n(sets, static_cast<std::size_t>(set_count32),
                  VK_NULL_HANDLE);
      SetVulkanLastError(adapter, "accel_vulkan_descriptor_unavailable");
      return false;
    }
    allocated += batch;
  }

  ::rund::detail::counter::Accumulate(adapter.descriptor_set_allocate_count,
                                      set_count);
  RecordVulkanDescriptorSetupNs(adapter, MonotonicNanoseconds() - begin);
  return true;
}

#endif

} // namespace rund::node::accel::detail
