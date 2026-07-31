#pragma once

#include "../../stencil.hpp"
#include "../../stencil/model.hpp"
#include "../adapter/api.hpp"
#include "../barrier.hpp"
#include "../collective/pipeline.hpp"
#include "../command.hpp"
#include "../descriptor.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
inline constexpr rund::kernel::u32 kStencilBlockSize = 256u;
inline constexpr std::uint32_t kStencilDescriptorCount = 3u;

struct VulkanStencilEncodeResources {
  VulkanAdapter *adapter = nullptr;
  rund::kernel::StencilPlan plan{};
  VulkanCollectivePipeline *pipeline = nullptr;
  VulkanBuffer params{};
  const VulkanBuffer *input = nullptr;
  const VulkanBuffer *output = nullptr;
  VulkanStorageBinding input_binding{};
  VulkanStorageBinding output_binding{};
  VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
};

void DestroyVulkanStencilEncodeResources(void *raw);
[[nodiscard]] std::string
VulkanStencilSource(rund::kernel::StencilOp op,
                    rund::kernel::StencilElement element,
                    rund::kernel::ComputeDomain domain);
[[nodiscard]] VulkanCollectivePipeline *
AcquireStencilPipeline(VulkanAdapter &adapter,
                       const rund::kernel::StencilDesc &desc,
                       rund::kernel::ComputeDomain domain);
[[nodiscard]] bool
CreateVulkanStencilDescriptorSet(VulkanAdapter &adapter,
                                 VulkanStencilEncodeResources &resources);
#endif

} // namespace rund::node::accel::detail
