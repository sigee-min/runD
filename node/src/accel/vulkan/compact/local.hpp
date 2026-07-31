#pragma once

#include "../../compact.hpp"
#include "../../primitive/block.hpp"
#include "../adapter/api.hpp"
#include "../barrier.hpp"
#include "../buffer/resident/model.hpp"
#include "../collective/pipeline.hpp"
#include "../command.hpp"
#include "../descriptor.hpp"
#include "../status.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
inline constexpr std::uint32_t kCompactDescriptorCount = 6u;
inline constexpr rund::kernel::u32 kCompactDispatchCount = 3u;

struct CompactParams {
  rund::kernel::u64 element_count = 0u;
  rund::kernel::u64 output_capacity = 0u;
};

enum class CompactStage : std::uint8_t {
  Classify,
  Prefix,
  Scatter,
};

struct VulkanCompactEncodeResources {
  VulkanAdapter *adapter = nullptr;
  rund::kernel::CompactPlan plan{};
  rund::kernel::u32 block_count = 0u;
  VulkanCollectivePipeline *classify_pipeline = nullptr;
  VulkanCollectivePipeline *prefix_pipeline = nullptr;
  VulkanCollectivePipeline *scatter_pipeline = nullptr;
  VulkanBuffer params{};
  VulkanBuffer counts{};
  VulkanBuffer offsets{};
  VulkanStatus status{};
  const VulkanBuffer *output = nullptr;
  VkDescriptorSet classify_set = VK_NULL_HANDLE;
  VkDescriptorSet prefix_set = VK_NULL_HANDLE;
  VkDescriptorSet scatter_set = VK_NULL_HANDLE;
};

void DestroyVulkanCompactEncodeResources(void *raw);
[[nodiscard]] std::string VulkanCompactSource(CompactStage stage);
[[nodiscard]] VulkanCollectivePipeline *
AcquireCompactPipeline(VulkanAdapter &adapter,
                       const rund::kernel::CompactDesc &desc,
                       CompactStage stage);
[[nodiscard]] bool
CreateVulkanCompactDescriptorSets(VulkanAdapter &adapter,
                                  VulkanCompactEncodeResources &resources,
                                  const VulkanResidentBufferResult &flags,
                                  const VulkanResidentBufferResult &output);
#endif

} // namespace rund::node::accel::detail
