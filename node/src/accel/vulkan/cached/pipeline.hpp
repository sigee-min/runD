#pragma once

#include "../adapter/api.hpp"
#include <cstdint>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
inline constexpr std::uint32_t kVulkanCachedPipelineInlineDescriptorCount = 4u;

[[nodiscard]] bool VulkanCachedPipelineMatches(
    const VulkanCachedPipeline& pipeline,
    const rund::kernel::LoweringArtifact& artifact,
    std::uint64_t source_hash,
    rund::kernel::u64 input_buffer_count,
    rund::kernel::u64 output_buffer_count) noexcept;

[[nodiscard]] VulkanCachedPipeline *AcquireVulkanCachedPipeline(
    VulkanAdapter &adapter, const rund::kernel::ComputePlan &plan,
    rund::kernel::LoweringArtifact artifact);

[[nodiscard]] bool CreateVulkanCachedPipeline(
    VulkanAdapter& adapter,
    const rund::kernel::ComputePlan& plan,
    rund::kernel::LoweringArtifact artifact,
    const VulkanShader& shader,
    VulkanCachedPipeline& pipeline);
#endif

}  // namespace rund::node::accel::detail
