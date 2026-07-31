#pragma once

#include "state.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

struct SortDispatch {
  rund::kernel::u32 base_block = 0u;
};

void DispatchVulkanSortChunks(const VulkanSortEncodeResources &sort,
                              VulkanCollectivePipeline &pipeline,
                              const VkCommandBuffer command) {
  for (rund::kernel::u32 chunk = 0u; chunk < sort.chunk_count; ++chunk) {
    const SortDispatch dispatch{chunk * sort.adapter->max_dispatch_groups};
    PushVulkanConstants(command, pipeline.pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(dispatch),
                        &dispatch);
    const VkDeviceSize offset =
        static_cast<VkDeviceSize>(chunk * 3u * sizeof(rund::kernel::u32));
    DispatchVulkanIndirect(command, sort.dispatch_args.buffer, offset);
  }
}

void EncodeVulkanSortClassify(const VulkanSortEncodeResources &sort,
                              const std::size_t pass,
                              const VkCommandBuffer command) {
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     sort.classify_pipeline->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        sort.classify_pipeline->pipeline_layout, 0u, 1u,
                        &sort.descriptors[pass].classify_set, 0u, nullptr);
  DispatchVulkanSortChunks(sort, *sort.classify_pipeline, command);
}

void EncodeVulkanSortClassifyBarrier(const VulkanSortEncodeResources &sort,
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
