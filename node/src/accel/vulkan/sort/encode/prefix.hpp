#pragma once

#include "../../../sort/block/bucket.hpp"
#include "classify.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

void EncodeVulkanSortPrefix(const VulkanSortEncodeResources &sort,
                            const std::size_t pass,
                            const VkCommandBuffer command) {
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     sort.prefix_pipeline->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        sort.prefix_pipeline->pipeline_layout, 0u, 1u,
                        &sort.descriptors[pass].prefix_set, 0u, nullptr);
  DispatchVulkan(command, static_cast<std::uint32_t>(kSortBucketCount), 1u, 1u);
}

void EncodeVulkanSortPrefixBarrier(const VulkanSortEncodeResources &sort,
                                   const VkCommandBuffer command) {
  const std::array<VkBufferMemoryBarrier, 2u> barriers{
      VulkanBufferBarrier(sort.block_counts, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT |
                              VK_ACCESS_SHADER_WRITE_BIT),
      VulkanBufferBarrier(sort.block_offsets, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT),
  };
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                       static_cast<std::uint32_t>(barriers.size()),
                       barriers.data(), 0u, nullptr);
}

void EncodeVulkanSortBase(const VulkanSortEncodeResources &sort,
                          const std::size_t pass,
                          const VkCommandBuffer command) {
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     sort.base_pipeline->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        sort.base_pipeline->pipeline_layout, 0u, 1u,
                        &sort.descriptors[pass].base_set, 0u, nullptr);
  DispatchVulkan(command, 1u, 1u, 1u);
}

void EncodeVulkanSortBaseBarrier(const VulkanSortEncodeResources &sort,
                                 const VkCommandBuffer command) {
  const VkBufferMemoryBarrier barrier = VulkanBufferBarrier(
      sort.block_counts, VK_ACCESS_SHADER_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                       1u, &barrier, 0u, nullptr);
}

} // namespace
#endif

} // namespace rund::node::accel::detail
