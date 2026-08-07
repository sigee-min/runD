#pragma once

#include "../../kernel/backend/run.hpp"
#include "../adapter/api.hpp"
#include "../adapter/buffer.hpp"
#include "../adapter/pipeline.hpp"
#include "../buffer/resident/model.hpp"
#include "../descriptor/binding.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanWindowParams final {
  std::uint64_t count_offset_words{};
  std::array<std::uint64_t, 3u> terminal_offset_words{};
  std::uint32_t maximum{};
  std::uint32_t tile{};
  std::uint32_t iteration{};
  std::uint32_t expected{};
  std::uint32_t state{};
  std::uint32_t has_terminal{};
  std::uint32_t command_count{};
  std::uint32_t phase{};
  std::uint32_t declared_step{};
  std::uint32_t overflow_reason{};
  std::uint32_t inner_bound{1u};
  std::uint32_t inner_advance{};
};

static_assert(sizeof(VulkanWindowParams) == 80u);

inline constexpr std::uint64_t VulkanGateParameterBytes = 12u;

struct VulkanWindowRoute final {
  VulkanResidentBufferResult count{};
  std::array<VulkanResidentBufferResult, 3u> terminals{};
  VulkanStorageBinding count_binding{};
  std::array<VulkanStorageBinding, 3u> terminal_bindings{};
  VulkanWindowParams params{};
  VkDescriptorSet descriptor{VK_NULL_HANDLE};
  std::uint32_t entry{};
};

struct VulkanGateRoute final {
  VulkanBuffer source{};
  VkDescriptorSet descriptor{VK_NULL_HANDLE};
};

struct VulkanWindowResources final {
  VulkanAdapter *adapter{};
  VulkanCollectivePipeline *pipeline{};
  VulkanCollectivePipeline *gate_pipeline{};
  VulkanBuffer states{};
  VulkanBuffer arguments{};
  VulkanBuffer original_arguments{};
  VulkanBuffer owners{};
  std::vector<VkDispatchIndirectCommand> original;
  std::vector<VulkanWindowRoute> routes;
  std::vector<VulkanGateRoute> gates;
  std::vector<VulkanCollectiveDescriptorLease> descriptor_leases;
  VulkanDispatchCapture capture{};
  std::uint64_t gate_capacity{};
  std::uint32_t state_count{};
};

[[nodiscard]] std::string_view VulkanWindowSourceText() noexcept;
[[nodiscard]] bool VulkanWindowSourceBytes(std::uint64_t &bytes) noexcept;
[[nodiscard]] std::string_view VulkanGateSourceText() noexcept;

[[nodiscard]] rund::AccelCheck PrepareVulkanWindow(
    VulkanAdapter &adapter, std::span<const BackendBatchEntry> entries,
    std::uint64_t dispatch_capacity, std::uint64_t gate_capacity,
    const PreparedPipelineStatusLayout &status, const VulkanBuffer &control,
    VulkanWindowResources &resources);

void DestroyVulkanWindow(VulkanWindowResources &resources) noexcept;

[[nodiscard]] bool EncodeVulkanWindow(VkCommandBuffer command,
                                      const VulkanWindowResources &resources,
                                      std::uint32_t entry,
                                      bool preflight = false) noexcept;
[[nodiscard]] bool
EncodeVulkanWindowStart(VkCommandBuffer command,
                        const VulkanWindowResources &resources) noexcept;
[[nodiscard]] bool
FreezeVulkanWindow(VulkanWindowResources &resources) noexcept;

[[nodiscard]] std::uint64_t
VulkanWindowHostBytes(const VulkanWindowResources &resources) noexcept;

#endif

} // namespace rund::node::accel::detail
