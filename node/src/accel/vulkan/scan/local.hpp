#pragma once

#include "../../scan/shape.hpp"
#include "../../scan/vulkan.hpp"
#include "../adapter/api.hpp"
#include "../barrier.hpp"
#include "../collective/chunk.hpp"
#include "../command.hpp"
#include "../descriptor.hpp"
#include "../status.hpp"
#include <array>
#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
inline constexpr std::uint32_t kScanDescriptorCount = 6u;
inline constexpr std::uint32_t kScanPushBytes = sizeof(rund::kernel::u64);

struct ScanParams {
  rund::kernel::u64 element_count = 0u;
  rund::kernel::u64 block_size = 0u;
  rund::kernel::u64 block_count = 0u;
  rund::kernel::u32 count_words = 0u;
  rund::kernel::u32 inclusive = 0u;
};
static_assert(sizeof(ScanParams) == 32u);

struct VulkanScanEncodeResources {
  VulkanAdapter *adapter = nullptr;
  VulkanCollectivePipeline *block = nullptr;
  VulkanCollectivePipeline *prefix = nullptr;
  VulkanCollectivePipeline *offset = nullptr;
  VulkanBuffer params{};
  VulkanBuffer output{};
  VulkanStorageBinding output_binding{};
  VulkanBuffer totals{};
  VulkanStatus *status = nullptr;
  VulkanStorageBinding logical_count{};
  VkDescriptorSet block_set = VK_NULL_HANDLE;
  VkDescriptorSet prefix_set = VK_NULL_HANDLE;
  VkDescriptorSet offset_set = VK_NULL_HANDLE;
  rund::kernel::u64 block_count = 0u;
  rund::kernel::u64 pass_count = 0u;
  rund::kernel::u64 dispatch_count = 0u;
};

void DestroyVulkanScanEncodeResources(void *raw);
[[nodiscard]] bool CreateVulkanScanDescriptorSets(
    VulkanAdapter &adapter, VulkanScanEncodeResources &resources,
    VulkanStorageBinding input, VulkanStorageBinding logical_count);
#endif

} // namespace rund::node::accel::detail
