#pragma once

#include <accel/check.hpp>

#include "state.hpp"

#include <algorithm>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

struct ScanDispatch {
  rund::kernel::u64 base_block = 0u;
};

void DispatchVulkanScanChunks(const VulkanScanEncodeResources &scan,
                              VulkanCollectivePipeline &pipeline,
                              const VkCommandBuffer command) {
  const rund::kernel::u64 limit = scan.adapter->max_dispatch_groups;
  for (rund::kernel::u64 base = 0u; base < scan.block_count;) {
    const ScanDispatch dispatch{base};
    const auto groups =
        static_cast<std::uint32_t>(std::min(limit, scan.block_count - base));
    PushVulkanConstants(command, pipeline.pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(dispatch),
                        &dispatch);
    DispatchVulkan(command, groups, 1u, 1u);
    base += groups;
  }
}

void EncodeVulkanScanBlocks(const VulkanScanEncodeResources &scan,
                            const VkCommandBuffer command) {
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     scan.block->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        scan.block->pipeline_layout, 0u, 1u, &scan.block_set,
                        0u, nullptr);
  DispatchVulkanScanChunks(scan, *scan.block, command);
}

void EncodeVulkanScanBlockBarrier(const VulkanScanEncodeResources &scan,
                                  const VkCommandBuffer command) {
  const std::array<VkBufferMemoryBarrier, 3u> barriers{
      VulkanBufferBarrier(scan.output, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT |
                              VK_ACCESS_SHADER_WRITE_BIT),
      VulkanBufferBarrier(scan.totals, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT |
                              VK_ACCESS_SHADER_WRITE_BIT),
      VulkanBufferBarrier(scan.status->device, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT |
                              VK_ACCESS_SHADER_WRITE_BIT),
  };
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                       static_cast<std::uint32_t>(barriers.size()),
                       barriers.data(), 0u, nullptr);
}

void EncodeVulkanScanPrefix(const VulkanScanEncodeResources &scan,
                            const VkCommandBuffer command) {
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     scan.prefix->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        scan.prefix->pipeline_layout, 0u, 1u, &scan.prefix_set,
                        0u, nullptr);
  DispatchVulkan(command, 1u, 1u, 1u);
}

void EncodeVulkanScanPrefixBarrier(const VulkanScanEncodeResources &scan,
                                   const VkCommandBuffer command) {
  const std::array<VkBufferMemoryBarrier, 2u> barriers{
      VulkanBufferBarrier(scan.totals, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT),
      VulkanBufferBarrier(scan.status->device, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT |
                              VK_ACCESS_SHADER_WRITE_BIT),
  };
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                       static_cast<std::uint32_t>(barriers.size()),
                       barriers.data(), 0u, nullptr);
}

void EncodeVulkanScanOffset(const VulkanScanEncodeResources &scan,
                            const VkCommandBuffer command) {
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     scan.offset->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        scan.offset->pipeline_layout, 0u, 1u, &scan.offset_set,
                        0u, nullptr);
  DispatchVulkanScanChunks(scan, *scan.offset, command);
}

} // namespace
#endif

} // namespace rund::node::accel::detail
