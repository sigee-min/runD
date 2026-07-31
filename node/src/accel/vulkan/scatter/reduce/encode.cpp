#include "model.hpp"

#include "../../barrier.hpp"
#include "../../collective/finish.hpp"

#include <array>
#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

rund::AccelCheck
EncodeVulkanScatterReduce(VulkanAdapter &adapter,
                          const std::shared_ptr<void> &resources,
                          void *const command_buffer) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const state =
      static_cast<VulkanScatterReduceResources *>(resources.get());
  const VkCommandBuffer command =
      reinterpret_cast<VkCommandBuffer>(command_buffer);
  if (state == nullptr || state->adapter != &adapter ||
      state->control_pipeline == nullptr || state->init_pipeline == nullptr ||
      state->fold_pipeline == nullptr || command == VK_NULL_HANDLE) {
    return {false, "compute_scatter_reduce_buffer_invalid"};
  }
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     state->control_pipeline->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        state->control_pipeline->pipeline_layout, 0u, 1u,
                        &state->control_descriptor, 0u, nullptr);
  DispatchVulkan(command, 1u, 1u, 1u);
  const std::array<VkBufferMemoryBarrier, 2u> control_barriers{
      VulkanBufferBarrier(state->indirect, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_INDIRECT_COMMAND_READ_BIT),
      VulkanBufferBarrier(state->status.device, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT |
                              VK_ACCESS_SHADER_WRITE_BIT)};
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                       0u, 0u, nullptr, control_barriers.size(),
                       control_barriers.data(), 0u, nullptr);
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     state->init_pipeline->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        state->init_pipeline->pipeline_layout, 0u, 1u,
                        &state->init_descriptor, 0u, nullptr);
  DispatchVulkanIndirect(command, state->indirect.buffer, 0u);
  const std::array<VkBufferMemoryBarrier, 2u> init_barriers{
      VulkanBufferBarrier(state->counts, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT |
                              VK_ACCESS_SHADER_WRITE_BIT),
      VulkanBufferBarrier(
          *state->output.device_buffer, VK_ACCESS_SHADER_WRITE_BIT,
          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT)};
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                       init_barriers.size(), init_barriers.data(), 0u, nullptr);
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     state->fold_pipeline->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        state->fold_pipeline->pipeline_layout, 0u, 1u,
                        &state->fold_descriptor, 0u, nullptr);
  DispatchVulkanIndirect(command, state->indirect.buffer,
                         3u * sizeof(std::uint32_t));
  const std::array<const VulkanBuffer *, 1u> outputs{
      state->output.device_buffer};
  if (!FinishVulkanStatus(command, state->status, outputs)) {
    return {false, "compute_scatter_reduce_buffer_invalid"};
  }
  return {true, "ok"};
#else
  (void)adapter;
  (void)resources;
  (void)command_buffer;
  return {false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
