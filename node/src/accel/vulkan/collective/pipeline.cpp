#include "pipeline.hpp"

#include "../cached/index.hpp"
#include "pipeline/acquire.hpp"

#include <kernel/core/checked.hpp>

#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
[[nodiscard]] bool GrowCollectiveDescriptorSets(
    VulkanAdapter &adapter, VulkanCollectivePipeline &pipeline,
    std::uint32_t descriptor_count, std::uint64_t set_count);

VulkanCollectivePipeline::~VulkanCollectivePipeline() { reset(); }

VulkanCollectivePipeline::VulkanCollectivePipeline(
    VulkanCollectivePipeline &&other) noexcept {
  *this = std::move(other);
}

VulkanCollectivePipeline &
VulkanCollectivePipeline::operator=(VulkanCollectivePipeline &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  reset();
  device = std::exchange(other.device, VK_NULL_HANDLE);
  key = other.key;
  descriptor_count = other.descriptor_count;
  push_bytes = other.push_bytes;
  source_hash = other.source_hash;
  source = std::move(other.source);
  specialization = other.specialization;
  descriptor_set_layout =
      std::exchange(other.descriptor_set_layout, VK_NULL_HANDLE);
  descriptor_pools = std::move(other.descriptor_pools);
  descriptor_sets = std::move(other.descriptor_sets);
  descriptor_leased = std::move(other.descriptor_leased);
  descriptor_epoch = other.descriptor_epoch;
  next_descriptor_slot = other.next_descriptor_slot;
  reusable_descriptor_count = other.reusable_descriptor_count;
  pipeline_layout = std::exchange(other.pipeline_layout, VK_NULL_HANDLE);
  pipeline = std::exchange(other.pipeline, VK_NULL_HANDLE);
  return *this;
}

void VulkanCollectivePipeline::reset() noexcept {
  if (device != VK_NULL_HANDLE) {
    if (pipeline != VK_NULL_HANDLE) {
      vkDestroyPipeline(device, pipeline, nullptr);
    }
    if (pipeline_layout != VK_NULL_HANDLE) {
      vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    }
    for (const VkDescriptorPool pool : descriptor_pools) {
      if (pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, pool, nullptr);
      }
    }
    if (descriptor_set_layout != VK_NULL_HANDLE) {
      vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
    }
  }
  device = VK_NULL_HANDLE;
  key = {};
  descriptor_count = 0u;
  push_bytes = 0u;
  source_hash = 0u;
  source.clear();
  specialization = {};
  descriptor_set_layout = VK_NULL_HANDLE;
  descriptor_pools.clear();
  descriptor_sets.clear();
  descriptor_leased.clear();
  descriptor_epoch = 0u;
  next_descriptor_slot = 0u;
  reusable_descriptor_count = 0u;
  pipeline_layout = VK_NULL_HANDLE;
  pipeline = VK_NULL_HANDLE;
}

void BeginVulkanCollectiveDescriptorEpoch(VulkanAdapter &adapter) noexcept {
  std::uint64_t &epoch = adapter.pipeline_index->descriptor_epoch;
  ++epoch;
  if (epoch != 0u) {
    return;
  }
  epoch = 1u;
  for (VulkanCollectivePipeline &pipeline : adapter.collective_pipelines) {
    pipeline.descriptor_epoch = 0u;
  }
}

void PrepareVulkanCollectiveDescriptorSlots(
    VulkanAdapter &adapter, VulkanCollectivePipeline &pipeline) noexcept {
  const std::uint64_t epoch = adapter.pipeline_index->descriptor_epoch;
  if (pipeline.descriptor_epoch == epoch) {
    return;
  }
  pipeline.descriptor_epoch = epoch;
  pipeline.next_descriptor_slot = 0u;
  pipeline.reusable_descriptor_count = pipeline.descriptor_sets.size();
}

bool ReserveVulkanCollectiveDescriptorDemand(
    VulkanAdapter &adapter, VulkanCollectivePipeline &pipeline,
    const std::uint32_t descriptor_count, const std::uint64_t set_count) {
  PrepareVulkanCollectiveDescriptorSlots(adapter, pipeline);
  if (descriptor_count != pipeline.descriptor_count) {
    SetVulkanLastError(adapter, "accel_vulkan_descriptor_unavailable");
    return false;
  }
  std::uint64_t available = 0u;
  for (std::size_t slot = pipeline.next_descriptor_slot;
       slot < pipeline.descriptor_sets.size(); ++slot) {
    available += pipeline.descriptor_leased[slot] == 0u ? 1u : 0u;
  }
  if (set_count <= available) {
    return true;
  }
  std::uint64_t target = 0u;
  return rund::kernel::checked::add(pipeline.descriptor_sets.size(),
                                    set_count - available, target) &&
         GrowCollectiveDescriptorSets(adapter, pipeline, descriptor_count,
                                      target);
}
#endif

} // namespace rund::node::accel::detail
