#pragma once

#include "../../stencil.hpp"
#include "../../stencil/model.hpp"
#include "../../stencil/shape.hpp"
#include "../adapter/api.hpp"
#include "../barrier.hpp"
#include "../collective/pipeline.hpp"
#include "../command.hpp"
#include "../descriptor.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
inline constexpr std::uint32_t kStencilDirectDescriptorCount = 3u;
inline constexpr std::uint32_t kStencilRangeDescriptorCount = 5u;

struct VulkanStencilEncodeResources {
  VulkanAdapter *adapter = nullptr;
  rund::kernel::StencilPlan plan{};
  RangeAggregatePlan range{
      RangeAggregatePlan::rejected("compute_range_aggregate_unavailable")};
  StencilGpuShape shape{};
  std::array<VulkanCollectivePipeline *, kRangeAggregateStageCapacity>
      pipelines{};
  std::array<VulkanBuffer, kRangeAggregateStageCapacity> params{};
  std::array<VkDescriptorSet, kRangeAggregateStageCapacity> descriptor_sets{};
  std::array<VulkanBuffer, kRangeTemporaryCapacity> temporaries{};
  std::uint32_t stage_count{};
  const VulkanBuffer *input = nullptr;
  const VulkanBuffer *output = nullptr;
  VulkanStorageBinding input_binding{};
  VulkanStorageBinding output_binding{};
};

void DestroyVulkanStencilEncodeResources(void *raw);
[[nodiscard]] std::string
VulkanStencilSource(rund::kernel::StencilOp op,
                    rund::kernel::StencilElement element,
                    rund::kernel::ComputeDomain domain, StencilGpuShape shape,
                    const RangeAggregatePlan &range);
[[nodiscard]] bool VulkanStencilSourceBytes(
    rund::kernel::StencilOp op, rund::kernel::StencilElement element,
    rund::kernel::ComputeDomain domain, StencilGpuShape shape,
    const RangeAggregatePlan &range, std::uint64_t &bytes) noexcept;
[[nodiscard]] bool VulkanStencilSourceMatches(
    rund::kernel::StencilOp op, rund::kernel::StencilElement element,
    rund::kernel::ComputeDomain domain, StencilGpuShape shape,
    const RangeAggregatePlan &range, std::string_view source,
    std::uint64_t source_hash) noexcept;
[[nodiscard]] VulkanCollectivePipeline *AcquireStencilPipeline(
    VulkanAdapter &adapter, const rund::kernel::StencilDesc &desc,
    rund::kernel::ComputeDomain domain, const RangeAggregatePlan &range);
[[nodiscard]] bool VulkanStencilPipelineMatches(
    const VulkanAdapter &adapter, const VulkanCollectivePipeline *pipeline,
    const rund::kernel::StencilDesc &desc, rund::kernel::ComputeDomain domain,
    const RangeAggregatePlan &range) noexcept;
[[nodiscard]] bool
CreateVulkanStencilDescriptorSet(VulkanAdapter &adapter,
                                 VulkanStencilEncodeResources &resources,
                                 std::uint32_t stage_index);
[[nodiscard]] bool VulkanStencilStageScratchBindings(
    const VulkanStencilEncodeResources &resources, std::uint32_t stage_index,
    const VulkanBuffer *&scratch0, const VulkanBuffer *&scratch1) noexcept;
#endif

} // namespace rund::node::accel::detail
