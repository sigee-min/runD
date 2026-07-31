#pragma once

#include "../../../scatter.hpp"

#include "../../adapter/api.hpp"
#include "../../buffer/resident/model.hpp"
#include "../../collective/pipeline.hpp"
#include "../../descriptor.hpp"
#include "../../status.hpp"

#include <memory>
#include <string>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

inline constexpr std::uint32_t kVulkanScatterReduceBindings = 8u;

enum class VulkanScatterReduceStage : std::uint64_t {
  Control = 1u,
  Init = 2u,
  Fold = 3u,
};

struct VulkanScatterReduceResources final {
  VulkanAdapter *adapter = nullptr;
  rund::kernel::ScatterReducePlan plan{};
  VulkanCollectivePipeline *control_pipeline = nullptr;
  VulkanCollectivePipeline *init_pipeline = nullptr;
  VulkanCollectivePipeline *fold_pipeline = nullptr;
  VulkanResidentBufferResult values{};
  VulkanResidentBufferResult indices{};
  VulkanResidentBufferResult count{};
  VulkanResidentBufferResult output{};
  VulkanBuffer params{};
  VulkanBuffer indirect{};
  VulkanBuffer counts{};
  VulkanStatus status{};
  VkDescriptorSet control_descriptor = VK_NULL_HANDLE;
  VkDescriptorSet init_descriptor = VK_NULL_HANDLE;
  VkDescriptorSet fold_descriptor = VK_NULL_HANDLE;
};

[[nodiscard]] std::string
VulkanScatterReduceSource(const rund::kernel::ScatterReducePlan &plan,
                          VulkanScatterReduceStage stage);

[[nodiscard]] VulkanCollectivePipeline *
AcquireVulkanScatterReducePipeline(VulkanAdapter &adapter,
                                   const rund::kernel::ScatterReducePlan &plan,
                                   VulkanScatterReduceStage stage);

void DestroyVulkanScatterReduce(void *raw);

#endif

} // namespace rund::node::accel::detail
