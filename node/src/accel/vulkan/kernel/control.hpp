#pragma once

#include "../../kernel/memory.hpp"
#include "../../kernel/status.hpp"
#include "../adapter/api.hpp"
#include "../adapter/pipeline.hpp"
#include "../descriptor.hpp"
#include "ops/model.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanPipelineCanonicalStatus final {
  VulkanPipelineStatusSource source{};
  VkDescriptorSet descriptor{VK_NULL_HANDLE};
  std::uint32_t first{};
  std::uint32_t active_program{};
};

struct VulkanPipelineControlResources final {
  VulkanAdapter *adapter{};
  VulkanBuffer arena{};
  VulkanBuffer summary{};
  // Eight field-major U64 lanes keep zero and no-overflow initialization to
  // two contiguous fills instead of one command per declared step.
  VulkanBuffer profile{};
  VulkanCollectivePipeline *canonical_pipeline{};
  VulkanCollectivePipeline *reduce_pipeline{};
  VkDescriptorSet reduce_descriptor{VK_NULL_HANDLE};
  std::vector<VulkanCollectiveDescriptorLease> descriptor_leases;
  std::uint32_t profile_step_count{};
  std::uint64_t command_count{};
};

[[nodiscard]] rund::AccelCheck PrepareVulkanPipelineControl(
    VulkanAdapter &adapter, std::span<VulkanPipelineCanonicalStatus> statuses,
    const PreparedPipelineStatusLayout &layout, bool profile_steps,
    VulkanPipelineControlResources &control, PreparedPipelineMemory &memory);

void DestroyVulkanPipelineControl(
    VulkanPipelineControlResources &control) noexcept;

[[nodiscard]] bool EncodeVulkanPipelineCanonicalStatus(
    VkCommandBuffer command, const VulkanPipelineControlResources &control,
    const VulkanPipelineCanonicalStatus &status) noexcept;

[[nodiscard]] bool FoldVulkanPipelineControl(
    VkCommandBuffer command, const VulkanPipelineControlResources &control,
    PreparedProgramStatusSlice slice, std::uint32_t declared_step) noexcept;

[[nodiscard]] bool OpenVulkanPipelineControl(
    VkCommandBuffer command,
    const VulkanPipelineControlResources &control) noexcept;

[[nodiscard]] bool FinishVulkanPipelineControl(
    VkCommandBuffer command, const VulkanPipelineControlResources &control,
    const PreparedPipelineStatusLayout &layout) noexcept;

[[nodiscard]] bool ResetVulkanPipelineProfile(
    VkCommandBuffer command,
    const VulkanPipelineControlResources &control) noexcept;

[[nodiscard]] bool PublishVulkanPipelineControl(
    VkCommandBuffer command,
    const VulkanPipelineControlResources &control) noexcept;

[[nodiscard]] bool
ReadVulkanPipelineControl(const VulkanPipelineControlResources &resources,
                          PreparedPipelineControl &control) noexcept;

[[nodiscard]] bool ReadVulkanPipelineProfile(
    const VulkanPipelineControlResources &resources,
    std::span<PreparedPipelineStepControl> controls) noexcept;

#endif

} // namespace rund::node::accel::detail
