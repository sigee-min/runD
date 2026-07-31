#pragma once

#include "../adapter/api.hpp"
#include <cstdint>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] VulkanCollectivePipeline *AcquireVulkanCollectivePipeline(
    VulkanAdapter &adapter, std::uint32_t descriptor_count,
    std::uint32_t push_bytes, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact,
    const VulkanSpecialization &specialization = {});

void BeginVulkanCollectiveDescriptorEpoch(VulkanAdapter &adapter) noexcept;
void PrepareVulkanCollectiveDescriptorSlots(
    VulkanAdapter &adapter, VulkanCollectivePipeline &pipeline) noexcept;

#endif

} // namespace rund::node::accel::detail
