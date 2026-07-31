#pragma once

#include "../../../sort.hpp"
#include "../../adapter/api.hpp"
#include "../../collective/pipeline.hpp"
#include "../../descriptor/binding.hpp"
#include "../../status.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
inline constexpr std::size_t kMaxSortPasses = 8u;
inline constexpr std::uint32_t kSortDescriptorCount = 9u;
inline constexpr std::uint32_t kSortPushBytes = sizeof(rund::kernel::u32);

struct SortParams {
  rund::kernel::u64 element_count = 0u;
  rund::kernel::u32 block_count = 0u;
  rund::kernel::u32 pass_index = 0u;
  rund::kernel::u32 identity_values = 0u;
  rund::kernel::u32 signed_order = 0u;
  rund::kernel::u32 pass_count = 0u;
  rund::kernel::u32 count_words = 0u;
  rund::kernel::u32 max_dispatch_groups = 0u;
  rund::kernel::u32 chunk_count = 0u;
};

enum class SortStage : std::uint8_t {
  Dispatch,
  Classify,
  Prefix,
  Base,
  Scatter
};

struct SortPassDescriptors {
  VkDescriptorSet classify_set = VK_NULL_HANDLE;
  VkDescriptorSet prefix_set = VK_NULL_HANDLE;
  VkDescriptorSet base_set = VK_NULL_HANDLE;
  VkDescriptorSet scatter_set = VK_NULL_HANDLE;
};

struct VulkanSortEncodeResources {
  VulkanAdapter *adapter = nullptr;
  rund::kernel::SortPlan plan{};
  rund::kernel::GraphControl control{};
  VulkanStorageBinding logical_count{};
  rund::kernel::u32 block_count = 0u;
  rund::kernel::u32 chunk_count = 0u;
  rund::kernel::u64 dispatch_count = 0u;
  VulkanCollectivePipeline *dispatch_pipeline = nullptr;
  VulkanCollectivePipeline *classify_pipeline = nullptr;
  VulkanCollectivePipeline *prefix_pipeline = nullptr;
  VulkanCollectivePipeline *base_pipeline = nullptr;
  VulkanCollectivePipeline *scatter_pipeline = nullptr;
  VulkanBuffer temp_keys{};
  VulkanBuffer temp_values{};
  VulkanBuffer block_counts{};
  VulkanBuffer block_offsets{};
  VulkanBuffer dispatch_args{};
  VulkanStatus status{};
  VulkanBuffer params{};
  VkDeviceSize params_stride{};
  VkDescriptorSet dispatch_descriptor = VK_NULL_HANDLE;
  std::array<SortPassDescriptors, kMaxSortPasses> descriptors{};
  std::array<const VulkanBuffer *, kMaxSortPasses> target_keys{};
  std::array<const VulkanBuffer *, kMaxSortPasses> target_values{};
  std::size_t pass_count = 0u;
};

struct VulkanSortPrepareShape {
  rund::kernel::u32 block_count = 0u;
  rund::kernel::u32 chunk_count = 0u;
  rund::kernel::u64 block_table_bytes = 0u;
  rund::kernel::u64 count_bytes = 0u;
};

struct VulkanSortResidentBuffers {
  VulkanStorageBinding read_keys{};
  VulkanStorageBinding read_values{};
  VulkanStorageBinding write_keys{};
  VulkanStorageBinding write_values{};
  VulkanStorageBinding logical_count{};
  bool identity_values = false;
};
#endif

} // namespace rund::node::accel::detail
