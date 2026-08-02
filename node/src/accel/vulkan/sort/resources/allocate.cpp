#include <accel/check.hpp>

#include "../local/api.hpp"

#include "../../collective/chunk.hpp"
#include "../../kernel/pipeline/template.hpp"

#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
rund::AccelCheck AllocateVulkanSortSharedResources(
    VulkanAdapter &adapter, const rund::kernel::SortDesc &desc,
    const rund::kernel::SortPlan &plan, const VulkanSortPrepareShape &shape,
    VulkanSortEncodeResources &resources,
    const VulkanKernelImmutablePipelines *const pipelines) {
  resources.plan = plan;
  resources.block_count = shape.block_count;
  resources.chunk_count = shape.chunk_count;
  resources.pass_count = static_cast<std::size_t>(plan.radix_pass_count);
  const VkDeviceSize alignment = adapter.storage_align;
  if (alignment == 0u ||
      sizeof(SortParams) >
          std::numeric_limits<VkDeviceSize>::max() - (alignment - 1u)) {
    return rund::AccelCheck{false, "compute_resident_bytes_invalid"};
  }
  resources.params_stride =
      ((sizeof(SortParams) + alignment - 1u) / alignment) * alignment;
  if (resources.pass_count >
      std::numeric_limits<VkDeviceSize>::max() / resources.params_stride) {
    return rund::AccelCheck{false, "compute_resident_bytes_invalid"};
  }
  resources.dispatch_count = SortDispatches(
      resources.pass_count, resources.block_count, adapter.max_dispatch_groups);
  resources.dispatch_pipeline =
      pipelines == nullptr
          ? AcquireSortPipeline(adapter, desc, SortStage::Dispatch)
          : pipelines->borrow(rund::kernel::NodeKind::Sort, 5u, 0u,
                              kSortDescriptorCount, 1u);
  resources.classify_pipeline =
      pipelines == nullptr
          ? AcquireSortPipeline(adapter, desc, SortStage::Classify)
          : pipelines->borrow(rund::kernel::NodeKind::Sort, 5u, 1u,
                              kSortDescriptorCount, plan.radix_pass_count);
  resources.prefix_pipeline =
      pipelines == nullptr
          ? AcquireSortPipeline(adapter, desc, SortStage::Prefix)
          : pipelines->borrow(rund::kernel::NodeKind::Sort, 5u, 2u,
                              kSortDescriptorCount, plan.radix_pass_count);
  resources.base_pipeline =
      pipelines == nullptr
          ? AcquireSortPipeline(adapter, desc, SortStage::Base)
          : pipelines->borrow(rund::kernel::NodeKind::Sort, 5u, 3u,
                              kSortDescriptorCount, plan.radix_pass_count);
  resources.scatter_pipeline =
      pipelines == nullptr
          ? AcquireSortPipeline(adapter, desc, SortStage::Scatter)
          : pipelines->borrow(rund::kernel::NodeKind::Sort, 5u, 4u,
                              kSortDescriptorCount, plan.radix_pass_count);
  if (resources.dispatch_count == 0u ||
      resources.dispatch_pipeline == nullptr ||
      resources.classify_pipeline == nullptr ||
      resources.prefix_pipeline == nullptr ||
      resources.base_pipeline == nullptr ||
      resources.scatter_pipeline == nullptr ||
      !ReserveVulkanCollectiveDescriptorSets(
          adapter, *resources.dispatch_pipeline, kSortDescriptorCount, 1u) ||
      !ReserveVulkanCollectiveDescriptorSets(
          adapter, *resources.classify_pipeline, kSortDescriptorCount,
          resources.pass_count) ||
      !ReserveVulkanCollectiveDescriptorSets(
          adapter, *resources.prefix_pipeline, kSortDescriptorCount,
          resources.pass_count) ||
      !ReserveVulkanCollectiveDescriptorSets(adapter, *resources.base_pipeline,
                                             kSortDescriptorCount,
                                             resources.pass_count) ||
      !ReserveVulkanCollectiveDescriptorSets(
          adapter, *resources.scatter_pipeline, kSortDescriptorCount,
          resources.pass_count) ||
      !CreateVulkanBuffer(adapter,
                          static_cast<VkDeviceSize>(resources.pass_count) *
                              resources.params_stride,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          resources.params) ||
      !CreateVulkanBuffer(
          adapter, static_cast<VkDeviceSize>(plan.temp_key_bytes),
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, resources.temp_keys, nullptr,
          VulkanMemoryUse::Scratch) ||
      !CreateVulkanBuffer(
          adapter, static_cast<VkDeviceSize>(plan.temp_value_bytes),
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, resources.temp_values, nullptr,
          VulkanMemoryUse::Scratch) ||
      !CreateVulkanBuffer(adapter, static_cast<VkDeviceSize>(shape.count_bytes),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          resources.block_counts, nullptr,
                          VulkanMemoryUse::Scratch) ||
      !CreateVulkanBuffer(
          adapter, static_cast<VkDeviceSize>(shape.block_table_bytes),
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, resources.block_offsets, nullptr,
          VulkanMemoryUse::Scratch) ||
      !CreateVulkanBuffer(adapter,
                          static_cast<VkDeviceSize>(3u * resources.chunk_count *
                                                    sizeof(rund::kernel::u32)),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                              VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                          resources.dispatch_args, nullptr,
                          VulkanMemoryUse::Device) ||
      !CreateVulkanStatus(adapter, sizeof(rund::kernel::u32),
                          resources.status)) {
    return rund::AccelCheck{false, VulkanLastError(&adapter)};
  }
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
