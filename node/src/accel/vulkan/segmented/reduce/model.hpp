#pragma once

#include "../../../segmented/reduce/model.hpp"

#include "../../adapter/api.hpp"
#include "../../collective/pipeline.hpp"
#include "../../descriptor.hpp"
#include "../../status.hpp"

#include <kernel/program/compute/segmented/reduce/model.hpp>

#include <memory>
#include <string>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

inline constexpr std::uint32_t kVulkanSegmentedReduceBindings = 6u;

struct VulkanSegmentedReduceParams final {
  rund::kernel::u64 count{};
  rund::kernel::u64 block_count{};
};

struct VulkanSegmentedReduceResources final {
  VulkanAdapter *adapter = nullptr;
  rund::kernel::SegmentedReducePlan plan{};
  VulkanCollectivePipeline *classify = nullptr;
  VulkanCollectivePipeline *prefix = nullptr;
  VulkanCollectivePipeline *scatter = nullptr;
  VulkanCollectivePipeline *reduce = nullptr;
  VulkanBuffer params{};
  VulkanBuffer block_counts{};
  VulkanBuffer block_offsets{};
  VulkanBuffer segment_starts{};
  VulkanBuffer segment_count{};
  VulkanBuffer dispatch_args{};
  VulkanStatus status{};
  const VulkanBuffer *output = nullptr;
  VulkanStorageBinding input_binding{};
  VulkanStorageBinding heads_binding{};
  VulkanStorageBinding output_binding{};
  VkDescriptorSet classify_set = VK_NULL_HANDLE;
  VkDescriptorSet prefix_set = VK_NULL_HANDLE;
  VkDescriptorSet scatter_set = VK_NULL_HANDLE;
  VkDescriptorSet reduce_set = VK_NULL_HANDLE;
};

void DestroyVulkanSegmentedReduce(void *raw);

[[nodiscard]] std::string VulkanSegmentedClassifySource();
[[nodiscard]] std::string VulkanSegmentedPrefixSource();
[[nodiscard]] std::string VulkanSegmentedScatterSource();
[[nodiscard]] std::string
VulkanSegmentedReduceSource(const rund::kernel::SegmentedReducePlan &plan,
                            rund::kernel::ComputeDomain domain);

[[nodiscard]] VulkanCollectivePipeline *AcquireVulkanSegmentedIndex(
    VulkanAdapter &adapter, const rund::kernel::SegmentedReduceDesc &desc,
    rund::kernel::ComputeDomain domain, const char *source);
[[nodiscard]] VulkanCollectivePipeline *
AcquireVulkanSegmentedReduce(VulkanAdapter &adapter,
                             const rund::kernel::SegmentedReduceDesc &desc,
                             const rund::kernel::SegmentedReducePlan &plan,
                             rund::kernel::ComputeDomain domain);

#endif

} // namespace rund::node::accel::detail
