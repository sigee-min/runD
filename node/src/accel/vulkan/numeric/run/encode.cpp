#include "../resource.hpp"

#include "../../barrier.hpp"
#include "../../status.hpp"

#include <kernel/program/compute/transform/stage.hpp>

#include <array>
#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

rund::AccelCheck EncodeVulkanNumeric(VulkanAdapter &adapter,
                                     const std::shared_ptr<void> &prepared,
                                     void *const command_buffer) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const state = static_cast<VulkanNumericPrepared *>(prepared.get());
  const VkCommandBuffer command =
      reinterpret_cast<VkCommandBuffer>(command_buffer);
  if (state == nullptr || state->adapter != &adapter ||
      state->pipeline == nullptr || state->descriptor == VK_NULL_HANDLE ||
      command == VK_NULL_HANDLE || state->groups == 0u) {
    return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
  }
  const VkDeviceSize status_bytes =
      state->status_count * sizeof(rund::kernel::u32);
  if (status_bytes != 0u &&
      !ResetVulkanStatus(command, state->status_binding, status_bytes)) {
    return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
  }
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     state->pipeline->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        state->pipeline->pipeline_layout, 0u, 1u,
                        &state->descriptor, 0u, nullptr);
  if (state->transform_count != 0u) {
    const auto dispatch = [&](const rund::kernel::transform_stage::Batch batch,
                              const std::uint32_t groups) {
      PushVulkanConstants(command, state->pipeline->pipeline_layout,
                          VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(batch),
                          &batch);
      DispatchVulkan(command, groups, 1u, 1u);
    };
    const auto barrier = [&] {
      std::array<VkBufferMemoryBarrier, 2u> stages{};
      for (std::size_t index = 0u; index < stages.size(); ++index) {
        stages[index] = VulkanBufferBarrier(
            *state->outputs[index], VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
      }
      vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u,
                           nullptr, static_cast<std::uint32_t>(stages.size()),
                           stages.data(), 0u, nullptr);
    };
    const auto local =
        rund::kernel::transform_stage::Describe(state->transform_count, 1u);
    dispatch(local,
             static_cast<std::uint32_t>(rund::kernel::transform_stage::Groups(
                 rund::kernel::transform_stage::Threads(state->transform_count,
                                                        local))));
    if (state->transform_count != 1u) {
      barrier();
      for (rund::kernel::u64 span =
               rund::kernel::transform_stage::FirstGlobalSpan;
           span != 0u && span <= state->transform_count;) {
        const auto batch = rund::kernel::transform_stage::Describe(
            state->transform_count, span);
        dispatch(batch, static_cast<std::uint32_t>(
                            rund::kernel::transform_stage::Groups(
                                rund::kernel::transform_stage::Threads(
                                    state->transform_count, batch))));
        span =
            rund::kernel::transform_stage::Next(state->transform_count, batch);
        if (span == 0u) {
          break;
        }
        barrier();
      }
    }
  } else {
    DispatchVulkan(command, state->groups, 1u, 1u);
  }
  std::array<VkBufferMemoryBarrier, 3u> barriers{};
  const VkAccessFlags output_access =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
      (state->pipeline_private ? 0u : VK_ACCESS_TRANSFER_READ_BIT);
  for (std::size_t index = 0u; index < state->output_count; ++index) {
    barriers[index] = VulkanBufferBarrier(
        *state->outputs[index], VK_ACCESS_SHADER_WRITE_BIT, output_access);
  }
  const VkPipelineStageFlags output_stages =
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
      (state->pipeline_private ? 0u : VK_PIPELINE_STAGE_TRANSFER_BIT);
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       output_stages, 0u, 0u, nullptr,
                       static_cast<std::uint32_t>(state->output_count),
                       barriers.data(), 0u, nullptr);
  if (status_bytes != 0u && !state->pipeline_private) {
    const VkBufferCopy copy{.srcOffset = state->status_binding.offset,
                            .size = status_bytes};
    vkCmdCopyBuffer(command, state->status->buffer,
                    state->status_readback.buffer, 1u, &copy);
    const VkBufferMemoryBarrier readable = VulkanBufferBarrier(
        state->status_readback, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_HOST_READ_BIT);
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0u, 0u, nullptr, 1u,
                         &readable, 0u, nullptr);
  }
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)prepared;
  (void)command_buffer;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
