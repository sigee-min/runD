#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "state.hpp"

#include "../../barrier.hpp"
#include "../../command.hpp"
#include "../../descriptor.hpp"

#include <string>

namespace rund::node::accel::detail {

struct VulkanKernelImmutablePipelines;

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
void DestroyVulkanSortEncodeResources(void *raw);
[[nodiscard]] rund::AccelCheck ValidateVulkanSortPrepareShape(
    VulkanAdapter &adapter, const rund::kernel::SortDesc &desc,
    const rund::kernel::SortPlan &plan, const SortBinds &bindings,
    VulkanSortPrepareShape &shape);
[[nodiscard]] rund::AccelCheck LookupVulkanSortResidentBuffers(
    VulkanAdapter &adapter, const rund::AccelDevice &pick,
    const rund::kernel::SortPlan &plan, const SortBinds &bindings,
    VulkanSortResidentBuffers &buffers);
[[nodiscard]] rund::AccelCheck AllocateVulkanSortSharedResources(
    VulkanAdapter &adapter, const rund::kernel::SortDesc &desc,
    const rund::kernel::SortPlan &plan, const VulkanSortPrepareShape &shape,
    VulkanSortEncodeResources &resources,
    const VulkanKernelImmutablePipelines *pipelines);
[[nodiscard]] rund::AccelCheck PrepareVulkanSortPass(
    VulkanAdapter &adapter, const rund::kernel::SortPlan &plan,
    const VulkanSortResidentBuffers &buffers, bool signed_order,
    std::size_t pass, VulkanSortEncodeResources &resources, SortParams &params);
[[nodiscard]] rund::AccelCheck
PrepareVulkanSortDispatch(VulkanAdapter &adapter,
                          const VulkanSortResidentBuffers &buffers,
                          VulkanSortEncodeResources &resources);
[[nodiscard]] bool
SortBlockShapeOk(const rund::kernel::SortPlan &plan,
                 rund::kernel::u32 &block_count,
                 rund::kernel::u64 &block_table_bytes) noexcept;
[[nodiscard]] std::string VulkanSortSource(rund::kernel::SortKey key,
                                           SortStage stage);
[[nodiscard]] bool VulkanSortSourceBytes(rund::kernel::SortKey key,
                                         SortStage stage,
                                         std::uint64_t &bytes) noexcept;
[[nodiscard]] VulkanCollectivePipeline *
AcquireSortPipeline(VulkanAdapter &adapter, const rund::kernel::SortDesc &desc,
                    SortStage stage);
#endif

} // namespace rund::node::accel::detail
