#pragma once

#include "../../reduce/pass.hpp"
#include "../adapter/api.hpp"
#include "../barrier.hpp"
#include "../collective/pipeline.hpp"
#include "../command.hpp"
#include "../descriptor.hpp"
#include "../status.hpp"
#include <kernel/program/compute/reduce/model.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
inline constexpr std::uint32_t kReduceDescriptorCount = 6u;

struct VulkanReduceEncodeResources {
  VulkanAdapter *adapter = nullptr;
  rund::kernel::ReducePlan plan{};
  VulkanCollectivePipeline *pipeline = nullptr;
  VulkanBuffer params{};
  VkDeviceSize params_stride = 0u;
  VulkanBuffer partial{};
  VulkanStatus status{};
  VulkanStorageBinding input{};
  VulkanStorageBinding output{};
  VulkanStorageBinding logical_count{};
  std::vector<VkDescriptorSet> descriptor_sets{};
};

void DestroyVulkanReduceEncodeResources(void *raw);
[[nodiscard]] bool
VulkanReduceIndexRangeOk(const rund::kernel::ReducePlan &plan) noexcept;
[[nodiscard]] std::string VulkanReduceSource(
    rund::kernel::ReduceOp op, rund::kernel::ReduceElement element,
    rund::kernel::u64 block_size, rund::kernel::ComputeDomain domain);
[[nodiscard]] VulkanCollectivePipeline *
AcquireReducePipeline(VulkanAdapter &adapter,
                      const rund::kernel::ReduceDesc &desc,
                      rund::kernel::ComputeDomain domain);
[[nodiscard]] bool CreateVulkanReducePassDescriptorSet(
    VulkanAdapter &adapter, VulkanReduceEncodeResources &resources,
    std::size_t index, VulkanStorageBinding read_buffer);
#endif

} // namespace rund::node::accel::detail
