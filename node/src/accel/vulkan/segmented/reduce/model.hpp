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

enum class VulkanSegmentedReduceStage : std::uint64_t {
  Classify = 1u,
  Prefix = 2u,
  Scatter = 3u,
  Reduce = 4u,
};

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

[[nodiscard]] std::string
VulkanSegmentedReduceSource(const rund::kernel::SegmentedReducePlan &plan,
                            rund::kernel::ComputeDomain domain,
                            VulkanSegmentedReduceStage stage);
[[nodiscard]] bool VulkanSegmentedReduceSourceBytes(
    const rund::kernel::SegmentedReducePlan &plan,
    rund::kernel::ComputeDomain domain, VulkanSegmentedReduceStage stage,
    std::uint64_t &bytes) noexcept;

[[nodiscard]] VulkanCollectivePipeline *
AcquireVulkanSegmentedReducePipeline(
    VulkanAdapter &adapter, const rund::kernel::SegmentedReduceDesc &desc,
    const rund::kernel::SegmentedReducePlan &plan,
    rund::kernel::ComputeDomain domain, VulkanSegmentedReduceStage stage);

#endif

} // namespace rund::node::accel::detail
