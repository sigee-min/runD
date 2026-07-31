#pragma once

#include "../../segmented/model.hpp"
#include "../../segmented/vulkan.hpp"
#include "../adapter/api.hpp"
#include "../barrier.hpp"
#include "../collective/chunk.hpp"
#include "../collective/pipeline.hpp"
#include "../command.hpp"
#include "../descriptor.hpp"
#include "../status.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
inline constexpr std::uint32_t kSegmentedScanDescriptorCount = 7u;
inline constexpr std::uint32_t kSegmentedScanPushBytes =
    sizeof(rund::kernel::u64);

struct VulkanSegmentedScanEncodeResources {
  VulkanAdapter *adapter = nullptr;
  rund::kernel::SegmentedScanPlan plan{};
  VulkanCollectivePipeline *block = nullptr;
  VulkanCollectivePipeline *prefix = nullptr;
  VulkanCollectivePipeline *offset = nullptr;
  VulkanBuffer params{};
  VulkanBuffer offsets{};
  VulkanBuffer first_heads{};
  VulkanStatus status{};
  const VulkanBuffer *input = nullptr;
  const VulkanBuffer *heads = nullptr;
  const VulkanBuffer *output = nullptr;
  VulkanStorageBinding input_binding{};
  VulkanStorageBinding heads_binding{};
  VulkanStorageBinding output_binding{};
  VkDescriptorSet block_set = VK_NULL_HANDLE;
  VkDescriptorSet prefix_set = VK_NULL_HANDLE;
  VkDescriptorSet offset_set = VK_NULL_HANDLE;
  rund::kernel::u64 dispatch_count = 0u;
};

void DestroyVulkanSegmentedScanEncodeResources(void *raw);
[[nodiscard]] std::string
VulkanSegmentedScanSource(rund::kernel::SegmentedScanElement element,
                          rund::kernel::ComputeDomain domain,
                          std::string_view phase);
[[nodiscard]] VulkanCollectivePipeline *AcquireSegmentedScanPipeline(
    VulkanAdapter &adapter, const rund::kernel::SegmentedScanDesc &desc,
    rund::kernel::ComputeDomain domain, std::string_view phase);
[[nodiscard]] bool CreateVulkanSegmentedScanDescriptorSet(
    VulkanAdapter &adapter, VulkanSegmentedScanEncodeResources &resources);
#endif

} // namespace rund::node::accel::detail
