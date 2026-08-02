#pragma once

#include "../../kernel/backend/run.hpp"
#include "../adapter/pipeline.hpp"
#include "../buffer/resident/model.hpp"
#include "../descriptor/binding.hpp"
#include "control.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanPipelinePublishParams final {
  std::uint64_t count{};
  std::array<std::uint64_t, 3u> source_offset_words{};
  std::array<std::uint64_t, 3u> source_stride_words{};
  std::uint64_t target_offset_words{};
  std::uint64_t target_stride_words{};
  std::uint32_t element_words{};
  std::uint32_t declared_step_count{};
  std::uint32_t state{};
  std::uint32_t final{};
  std::uint32_t stop{};
  std::uint32_t maximum{};
  std::uint32_t tile{};
  std::uint32_t outer{};
  std::uint32_t kind{};
  std::uint64_t count_offset_words{};
};

static_assert(sizeof(VulkanPipelinePublishParams) == 120u);

struct VulkanPipelinePublishRoute final {
  std::array<VulkanResidentBufferResult, 3u> sources{};
  VulkanResidentBufferResult target{};
  VulkanResidentBufferResult count{};
  std::array<VulkanStorageBinding, 3u> source_bindings{};
  VulkanStorageBinding target_binding{};
  VulkanStorageBinding count_binding{};
  VulkanPipelinePublishParams params{};
  VkDescriptorSet descriptor{VK_NULL_HANDLE};
  VkDescriptorSet canonical_descriptor{VK_NULL_HANDLE};
  std::uint32_t groups_x{};
  std::uint32_t groups_y{};
};

struct VulkanPipelinePublishResources final {
  VulkanAdapter *adapter{};
  VulkanCollectivePipeline *pipeline{};
  std::vector<VulkanPipelinePublishRoute> routes;
  std::vector<VulkanCollectiveDescriptorLease> descriptor_leases;
};

struct VulkanWindowResources;

[[nodiscard]] std::string_view VulkanPublishSourceText() noexcept;

[[nodiscard]] rund::AccelCheck
PrepareVulkanPipelinePublish(VulkanAdapter &adapter,
                             std::span<const BackendPublish> publications,
                             const PreparedPipelineStatusLayout &status,
                             const VulkanPipelineControlResources &control,
                             const VulkanWindowResources &window,
                             VulkanPipelinePublishResources &resources);

void DestroyVulkanPipelinePublish(
    VulkanPipelinePublishResources &resources) noexcept;

[[nodiscard]] bool EncodeVulkanPipelinePublish(
    VkCommandBuffer command,
    const VulkanPipelinePublishResources &resources) noexcept;
[[nodiscard]] bool EncodeVulkanPipelineCanonicalize(
    VkCommandBuffer command, const VulkanPipelinePublishResources &resources,
    std::uint32_t state) noexcept;
[[nodiscard]] bool EncodeVulkanPipelineWindowPublish(
    VkCommandBuffer command, const VulkanPipelinePublishResources &resources,
    std::uint32_t state, std::uint32_t outer) noexcept;
[[nodiscard]] std::uint64_t VulkanPipelinePublishHostBytes(
    const VulkanPipelinePublishResources &resources) noexcept;

#endif

} // namespace rund::node::accel::detail
