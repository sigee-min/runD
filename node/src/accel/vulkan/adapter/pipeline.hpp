#pragma once

#include <kernel/program/compute/artifact.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
#include <vulkan/vulkan.h>
#endif

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanSpecialization final {
  std::array<std::uint32_t, 4u> values{};
  std::uint32_t count = 0u;

  [[nodiscard]] friend constexpr bool
  operator==(const VulkanSpecialization &,
             const VulkanSpecialization &) = default;
};

struct VulkanCachedPipeline {
  VulkanCachedPipeline() noexcept = default;
  ~VulkanCachedPipeline();
  VulkanCachedPipeline(const VulkanCachedPipeline &) = delete;
  VulkanCachedPipeline &operator=(const VulkanCachedPipeline &) = delete;
  VulkanCachedPipeline(VulkanCachedPipeline &&other) noexcept;
  VulkanCachedPipeline &operator=(VulkanCachedPipeline &&other) noexcept;

  void reset() noexcept;

  VkDevice device = VK_NULL_HANDLE;
  rund::kernel::ArtifactKey key{};
  std::uint64_t source_hash = 0u;
  std::string source{};
  rund::kernel::u64 input_buffer_count = 0u;
  rund::kernel::u64 output_buffer_count = 0u;
  std::uint64_t shader_hash = 0u;
  VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
  VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
};

struct VulkanCollectivePipeline {
  VulkanCollectivePipeline() noexcept = default;
  ~VulkanCollectivePipeline();
  VulkanCollectivePipeline(const VulkanCollectivePipeline &) = delete;
  VulkanCollectivePipeline &
  operator=(const VulkanCollectivePipeline &) = delete;
  VulkanCollectivePipeline(VulkanCollectivePipeline &&other) noexcept;
  VulkanCollectivePipeline &
  operator=(VulkanCollectivePipeline &&other) noexcept;

  void reset() noexcept;

  VkDevice device = VK_NULL_HANDLE;
  rund::kernel::ArtifactKey key{};
  std::uint32_t descriptor_count = 0u;
  std::uint32_t push_bytes = 0u;
  std::uint64_t source_hash = 0u;
  std::string source{};
  VulkanSpecialization specialization{};
  VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
  std::vector<VkDescriptorPool> descriptor_pools{};
  std::vector<VkDescriptorSet> descriptor_sets{};
  // Byte-addressable lease ownership keeps capacity/accounting exact and
  // avoids the proxy/bit-packing semantics of vector<bool>.
  std::vector<std::uint8_t> descriptor_leased{};
  std::uint64_t descriptor_epoch = 0u;
  std::uint64_t next_descriptor_slot = 0u;
  std::uint64_t reusable_descriptor_count = 0u;
  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
};

struct VulkanCollectiveDescriptorLease {
  VulkanCollectivePipeline *pipeline = nullptr;
  std::size_t slot = 0u;
};

#endif // defined(RUND_NODE_HAVE_VULKAN_SDK)

} // namespace rund::node::accel::detail
