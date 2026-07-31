#pragma once

#include "state.hpp"

#include <kernel/program/compute/reduce/plan.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

void EncodeVulkanReducePipeline(const VulkanReduceEncodeResources &reduce,
                                const VkCommandBuffer command) {
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     reduce.pipeline->pipeline);
}

void EncodeVulkanReduceGroups(const VulkanReduceEncodeResources &reduce,
                              const VkCommandBuffer command,
                              const std::size_t pass,
                              const std::uint32_t groups) {
  const VkDescriptorSet set = reduce.descriptor_sets[pass];
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        reduce.pipeline->pipeline_layout, 0u, 1u, &set, 0u,
                        nullptr);
  DispatchVulkan(command, groups, 1u, 1u);
}

[[nodiscard]] rund::kernel::u64 EncodeVulkanReducePass(
    const VulkanReduceEncodeResources &reduce, const VkCommandBuffer command,
    const rund::kernel::u64 current, const rund::kernel::u64 pass) {
  const rund::kernel::u64 next =
      rund::kernel::ReduceGroupCount(current, reduce.plan.block_size);
  EncodeVulkanReduceGroups(reduce, command, static_cast<std::size_t>(pass),
                           static_cast<std::uint32_t>(next));
  return next;
}

void EncodeVulkanReducePartialBarrier(const VulkanReduceEncodeResources &reduce,
                                      const VkCommandBuffer command) {
  const VkBufferMemoryBarrier barrier = VulkanBufferBarrier(
      reduce.partial, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                       1u, &barrier, 0u, nullptr);
}

} // namespace
#endif

} // namespace rund::node::accel::detail
