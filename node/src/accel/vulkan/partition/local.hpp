#pragma once

#include "../../partition.hpp"
#include "../../partition/model.hpp"
#include "../../primitive/block.hpp"
#include "../../scan/vulkan.hpp"
#include "../adapter/api.hpp"
#include "../barrier.hpp"
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
inline constexpr rund::kernel::u32 kPartitionBlockSize = 256u;
inline constexpr std::uint32_t kPartitionClassifyDescriptorCount = 3u;
inline constexpr std::uint32_t kPartitionScatterDescriptorCount = 5u;

enum class PartitionStage : std::uint8_t {
  Classify,
  Scatter,
};

struct VulkanPartitionEncodeResources {
  VulkanAdapter *adapter = nullptr;
  rund::kernel::PartitionPlan plan{};
  rund::kernel::ScanDesc scan_desc{};
  rund::kernel::ScanPlan scan_plan{};
  VulkanCollectivePipeline *classify_pipeline = nullptr;
  VulkanCollectivePipeline *scatter_pipeline = nullptr;
  VulkanBuffer params{};
  VulkanBuffer false_bits{};
  VulkanBuffer false_offsets{};
  VulkanBuffer false_totals{};
  VulkanStatus false_status{};
  const VulkanBuffer *flags = nullptr;
  const VulkanBuffer *values = nullptr;
  const VulkanBuffer *output = nullptr;
  rund::kernel::ResidentBufferRef flags_ref{};
  rund::kernel::ResidentBufferRef values_ref{};
  rund::kernel::ResidentBufferRef output_ref{};
  std::shared_ptr<void> false_scan_resources{};
  VkDescriptorSet classify_set = VK_NULL_HANDLE;
  VkDescriptorSet scatter_set = VK_NULL_HANDLE;
};

void DestroyVulkanPartitionEncodeResources(void *raw);
[[nodiscard]] std::string VulkanPartitionSource(PartitionStage stage,
                                                rund::kernel::u32 flag_bytes,
                                                rund::kernel::u32 value_bytes);
[[nodiscard]] VulkanCollectivePipeline *
AcquirePartitionPipeline(VulkanAdapter &adapter,
                         const rund::kernel::PartitionDesc &desc,
                         PartitionStage stage);
[[nodiscard]] bool
CreateVulkanPartitionDescriptorSets(VulkanAdapter &adapter,
                                    VulkanPartitionEncodeResources &resources);
#endif

} // namespace rund::node::accel::detail
