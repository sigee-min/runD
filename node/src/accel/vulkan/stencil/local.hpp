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
inline constexpr std::uint32_t kStencilDescriptorCount = 3u;

struct VulkanStencilEncodeResources {
  VulkanAdapter *adapter = nullptr;
  rund::kernel::StencilPlan plan{};
  StencilGpuShape shape{};
  VulkanCollectivePipeline *pipeline = nullptr;
  VulkanBuffer params{};
  const VulkanBuffer *input = nullptr;
  const VulkanBuffer *output = nullptr;
  VulkanStorageBinding input_binding{};
  VulkanStorageBinding output_binding{};
  VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
};

void DestroyVulkanStencilEncodeResources(void *raw);
[[nodiscard]] StencilGpuShape SelectVulkanStencilGpuShape(
    const VulkanAdapter &adapter, rund::kernel::u64 element_count,
    rund::kernel::u64 radius, rund::kernel::StencilElement element) noexcept;
[[nodiscard]] std::string
VulkanStencilSource(rund::kernel::StencilOp op,
                    rund::kernel::StencilElement element,
                    rund::kernel::ComputeDomain domain, StencilGpuShape shape);
[[nodiscard]] bool
VulkanStencilSourceBytes(rund::kernel::StencilOp op,
                         rund::kernel::StencilElement element,
                         rund::kernel::ComputeDomain domain,
                         StencilGpuShape shape, std::uint64_t &bytes) noexcept;
[[nodiscard]] bool VulkanStencilSourceMatches(
    rund::kernel::StencilOp op, rund::kernel::StencilElement element,
    rund::kernel::ComputeDomain domain, StencilGpuShape shape,
    std::string_view source, std::uint64_t source_hash) noexcept;
[[nodiscard]] VulkanCollectivePipeline *AcquireStencilPipeline(
    VulkanAdapter &adapter, const rund::kernel::StencilDesc &desc,
    rund::kernel::ComputeDomain domain, StencilGpuShape shape);
[[nodiscard]] bool VulkanStencilPipelineMatches(
    const VulkanAdapter &adapter, const VulkanCollectivePipeline *pipeline,
    const rund::kernel::StencilDesc &desc, rund::kernel::ComputeDomain domain,
    StencilGpuShape shape) noexcept;
[[nodiscard]] bool
CreateVulkanStencilDescriptorSet(VulkanAdapter &adapter,
                                 VulkanStencilEncodeResources &resources);
#endif

} // namespace rund::node::accel::detail
