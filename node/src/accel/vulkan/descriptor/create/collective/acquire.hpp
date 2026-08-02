#pragma once
#include "../../../collective/pipeline.hpp"
#include "grow.hpp"
namespace rund::node::accel::detail {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)

bool AcquireVulkanCollectiveDescriptorSet(
    VulkanAdapter& adapter,
    VulkanCollectivePipeline& pipeline,
    const std::uint32_t descriptor_count,
    VkDescriptorSet& set) {
  set = VK_NULL_HANDLE;
  if (adapter.active_descriptor_leases != nullptr &&
      adapter.active_descriptor_leases->size() ==
          adapter.active_descriptor_leases->capacity()) {
    SetVulkanLastError(adapter, "compute_pipeline_capacity");
    return false;
  }
  PrepareVulkanCollectiveDescriptorSlots(adapter, pipeline);
  std::uint64_t slot = pipeline.next_descriptor_slot;
  while (slot < pipeline.descriptor_leased.size() &&
         pipeline.descriptor_leased[static_cast<std::size_t>(slot)]) {
    ++slot;
  }
  if (!CollectiveDescriptorSlotOk(adapter, pipeline, descriptor_count, slot)) {
    return false;
  }
  const auto index = static_cast<std::size_t>(slot);
  if (!ReuseCollectiveDescriptorSet(adapter, pipeline, index, set)) {
    if (!GrowCollectiveDescriptorSets(adapter, pipeline, descriptor_count,
                                      slot + 1u)) {
      return false;
    }
    set = pipeline.descriptor_sets[index];
    if (set == VK_NULL_HANDLE) { return false; }
  }
  pipeline.next_descriptor_slot = slot + 1u;
  if (adapter.active_descriptor_leases != nullptr) {
    adapter.active_descriptor_leases->push_back(
        VulkanCollectiveDescriptorLease{&pipeline, index});
    pipeline.descriptor_leased[index] = true;
  }
  return true;
}

#endif

}  // namespace rund::node::accel::detail
