#include "model.hpp"

#include "../../../segmented/reduce/vulkan.hpp"

#include "../../barrier.hpp"
#include "../../collective/finish.hpp"

#include <array>

namespace rund::node::accel::detail {

rund::AccelCheck
EncodeVulkanSegmentedReduce(VulkanAdapter &adapter,
                            const std::shared_ptr<void> &resources,
                            void *const command_buffer) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const state =
      static_cast<VulkanSegmentedReduceResources *>(resources.get());
  const VkCommandBuffer command =
      reinterpret_cast<VkCommandBuffer>(command_buffer);
  if (state == nullptr || state->adapter != &adapter ||
      state->classify == nullptr || state->prefix == nullptr ||
      state->scatter == nullptr || state->reduce == nullptr ||
      command == VK_NULL_HANDLE ||
      state->status.device.buffer == VK_NULL_HANDLE) {
    return {false, "compute_segmented_reduce_invalid"};
  }
  if (!ResetVulkanStatus(command, state->status, sizeof(rund::kernel::u32))) {
    return {false, "compute_segmented_reduce_invalid"};
  }
  const SegmentedReduceLayout layout =
      SegmentedReduceLayoutFor(state->plan.element_count);
  const auto bind = [&](VulkanCollectivePipeline &pipeline,
                        const VkDescriptorSet set) {
    BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                       pipeline.pipeline);
    BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipeline.pipeline_layout, 0u, 1u, &set, 0u, nullptr);
  };
  bind(*state->classify, state->classify_set);
  DispatchVulkan(command, static_cast<rund::kernel::u32>(layout.index_groups),
                 1u, 1u);
  const std::array<VkBufferMemoryBarrier, 2u> classify_barriers{
      VulkanBufferBarrier(state->block_counts, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT),
      VulkanBufferBarrier(state->status.device, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT |
                              VK_ACCESS_SHADER_WRITE_BIT),
  };
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                       classify_barriers.size(), classify_barriers.data(), 0u,
                       nullptr);
  bind(*state->prefix, state->prefix_set);
  DispatchVulkan(command, 1u, 1u, 1u);
  const std::array<VkBufferMemoryBarrier, 3u> prefix_barriers{
      VulkanBufferBarrier(state->block_offsets, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT),
      VulkanBufferBarrier(state->segment_count, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT),
      VulkanBufferBarrier(state->dispatch_args, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_INDIRECT_COMMAND_READ_BIT),
  };
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                       0u, 0u, nullptr, prefix_barriers.size(),
                       prefix_barriers.data(), 0u, nullptr);
  bind(*state->scatter, state->scatter_set);
  DispatchVulkan(command, static_cast<rund::kernel::u32>(layout.index_groups),
                 1u, 1u);
  const VkBufferMemoryBarrier scatter_barrier =
      VulkanBufferBarrier(state->segment_starts, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT);
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                       1u, &scatter_barrier, 0u, nullptr);
  bind(*state->reduce, state->reduce_set);
  DispatchVulkanIndirect(command, state->dispatch_args.buffer, 0u);
  const std::array<const VulkanBuffer *, 1u> outputs{state->output};
  return FinishVulkanStatus(command, state->status, outputs)
             ? rund::AccelCheck{true, "ok"}
             : rund::AccelCheck{false, "compute_segmented_reduce_invalid"};
#else
  (void)adapter;
  (void)resources;
  (void)command_buffer;
  return {false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
