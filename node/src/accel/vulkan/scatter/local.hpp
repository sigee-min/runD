#pragma once

#include "../../scatter.hpp"
#include "../../scatter/model.hpp"
#include "../adapter/api.hpp"
#include "../barrier.hpp"
#include "../collective/pipeline.hpp"
#include "../command.hpp"
#include "../descriptor.hpp"
#include "../status.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
inline constexpr rund::kernel::u32 kScatterBlockSize = 256u;
inline constexpr std::uint32_t kScatterDescriptorCount = 5u;

struct VulkanScatterEncodeResources {
  VulkanAdapter *adapter = nullptr;
  rund::kernel::ScatterPlan plan{};
  VulkanCollectivePipeline *pipeline = nullptr;
  VulkanBuffer params{};
  VulkanStatus status{};
  const VulkanBuffer *values = nullptr;
  const VulkanBuffer *indices = nullptr;
  const VulkanBuffer *output = nullptr;
  VulkanStorageBinding values_binding{};
  VulkanStorageBinding indices_binding{};
  VulkanStorageBinding output_binding{};
  VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
};

void DestroyVulkanScatterEncodeResources(void *raw);
[[nodiscard]] std::string
VulkanScatterSource(rund::kernel::ScatterElement element);
[[nodiscard]] VulkanCollectivePipeline *
AcquireScatterPipeline(VulkanAdapter &adapter,
                       const rund::kernel::ScatterDesc &desc);
[[nodiscard]] bool
CreateVulkanScatterDescriptorSet(VulkanAdapter &adapter,
                                 VulkanScatterEncodeResources &resources);
#endif

} // namespace rund::node::accel::detail
