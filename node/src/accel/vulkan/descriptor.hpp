#pragma once

#include "adapter/api.hpp"
#include "cached/pipeline.hpp"
#include "collective/pipeline.hpp"
#include "descriptor/binding.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool AcquireVulkanCollectiveDescriptorSet(
    VulkanAdapter &adapter, VulkanCollectivePipeline &pipeline,
    std::uint32_t descriptor_count, VkDescriptorSet &set);

[[nodiscard]] bool ReserveVulkanCollectiveDescriptorSets(
    VulkanAdapter &adapter, VulkanCollectivePipeline &pipeline,
    std::uint32_t descriptor_count, std::uint64_t set_count);

[[nodiscard]] bool
CreateVulkanStorageDescriptorSet(VulkanAdapter &adapter,
                                 const VulkanCachedPipeline &pipeline,
                                 std::uint32_t descriptor_count,
                                 VkDescriptorPool &pool, VkDescriptorSet &set);

[[nodiscard]] bool CreateVulkanStorageDescriptorSets(
    VulkanAdapter &adapter, const VulkanCachedPipeline &pipeline,
    std::uint64_t set_count, VkDescriptorPool &pool,
    std::vector<VkDescriptorSet> &sets);

[[nodiscard]] bool ValidStorage(const VulkanAdapter &adapter,
                                const VkDescriptorBufferInfo &info) noexcept;

[[nodiscard]] bool PlanStorage(const VulkanAdapter &adapter,
                               const rund::kernel::ResidentBufferRef &ref,
                               std::uint64_t begin, std::uint64_t count,
                               StorageRange &range) noexcept;

[[nodiscard]] bool PlanStoragePage(const VulkanAdapter &adapter,
                                   const rund::kernel::ResidentBufferRef &ref,
                                   std::uint64_t begin,
                                   StorageRange &range) noexcept;

[[nodiscard]] bool
WriteVulkanStorageDescriptorSet(VulkanAdapter &adapter, VkDescriptorSet set,
                                const VulkanBuffer *const *buffers,
                                std::uint32_t descriptor_count);

[[nodiscard]] bool
WriteVulkanStorageDescriptorSet(VulkanAdapter &adapter, VkDescriptorSet set,
                                const VulkanStorageBinding *bindings,
                                std::uint32_t descriptor_count);

[[nodiscard]] bool
WriteVulkanStorageDescriptorSet(VulkanAdapter &adapter, VkDescriptorSet set,
                                const VkDescriptorBufferInfo *infos,
                                std::uint32_t descriptor_count);

template <std::size_t Count>
[[nodiscard]] bool WriteVulkanStorageDescriptorSet(
    VulkanAdapter &adapter, const VkDescriptorSet set,
    const std::array<const VulkanBuffer *, Count> &buffers) {
  return WriteVulkanStorageDescriptorSet(adapter, set, buffers.data(),
                                         static_cast<std::uint32_t>(Count));
}

template <std::size_t Count>
[[nodiscard]] bool WriteVulkanStorageDescriptorSet(
    VulkanAdapter &adapter, const VkDescriptorSet set,
    const std::array<VulkanStorageBinding, Count> &bindings) {
  return WriteVulkanStorageDescriptorSet(adapter, set, bindings.data(),
                                         static_cast<std::uint32_t>(Count));
}

#endif

} // namespace rund::node::accel::detail
