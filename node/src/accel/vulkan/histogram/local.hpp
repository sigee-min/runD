#pragma once

#include "../../histogram.hpp"
#include "../../histogram/model.hpp"
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
inline constexpr rund::kernel::u32 kHistogramBlockSize = 256u;
inline constexpr std::uint32_t kHistogramDescriptorCount = 4u;

struct VulkanHistogramEncodeResources {
  VulkanAdapter *adapter = nullptr;
  rund::kernel::HistogramPlan plan{};
  VulkanCollectivePipeline *clear_pipeline = nullptr;
  VulkanCollectivePipeline *count_pipeline = nullptr;
  VulkanBuffer params{};
  VulkanStatus status{};
  const VulkanBuffer *bins = nullptr;
  const VulkanBuffer *counts = nullptr;
  VulkanStorageBinding bins_binding{};
  VulkanStorageBinding counts_binding{};
  VkDescriptorSet clear_descriptor_set = VK_NULL_HANDLE;
  VkDescriptorSet count_descriptor_set = VK_NULL_HANDLE;
};

void DestroyVulkanHistogramEncodeResources(void *raw);
[[nodiscard]] std::string VulkanHistogramSource(bool clear);
[[nodiscard]] bool VulkanHistogramSourceBytes(bool clear,
                                              std::uint64_t &bytes) noexcept;
[[nodiscard]] VulkanCollectivePipeline *
AcquireHistogramPipeline(VulkanAdapter &adapter,
                         const rund::kernel::HistogramDesc &desc, bool clear);
[[nodiscard]] bool
CreateVulkanHistogramDescriptorSets(VulkanAdapter &adapter,
                                    VulkanHistogramEncodeResources &resources);
#endif

} // namespace rund::node::accel::detail
