#include "model.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

#include "../../command/dispatch.hpp"
#include "../../barrier.hpp"
#include "../../descriptor.hpp"

#include <array>

namespace rund::node::accel::detail {

bool EncodeVulkanPipelineCanonicalStatus(
    const VkCommandBuffer command,
    const VulkanPipelineControlResources &control,
    const VulkanPipelineCanonicalStatus &status) noexcept {
  if (command == VK_NULL_HANDLE || control.canonical_pipeline == nullptr ||
      status.descriptor == VK_NULL_HANDLE) {
    return false;
  }
  const VulkanCanonicalStatusParams params{
      .first = status.first,
      .count = status.source.count,
      .rule = static_cast<std::uint32_t>(status.source.rule),
      .success = status.source.success,
      .mapping_count = status.source.mapping_count,
      .invalid_reason =
          static_cast<std::uint32_t>(rund::compute::Reason::ReasonInvalid),
      .raw_values = status.source.raw_values,
      .reasons = status.source.reasons,
  };
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     control.canonical_pipeline->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        control.canonical_pipeline->pipeline_layout, 0u, 1u,
                        &status.descriptor, 0u, nullptr);
  PushVulkanConstants(command, control.canonical_pipeline->pipeline_layout,
                      VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(params), &params);
  const std::uint32_t groups =
      (status.source.count + VulkanPipelineControlThreads - 1u) /
      VulkanPipelineControlThreads;
  DispatchVulkan(command, groups, 1u, 1u);
  return true;
}

namespace {

[[nodiscard]] bool
EncodeVulkanPipelineControl(const VkCommandBuffer command,
                            const VulkanPipelineControlResources &control,
                            VulkanPipelineControlParams params) noexcept {
  rund::compute::PipelineNestedPhase failed_phase{};
  if (command == VK_NULL_HANDLE || control.reduce_pipeline == nullptr ||
      control.reduce_descriptor == VK_NULL_HANDLE ||
      control.summary.buffer == VK_NULL_HANDLE ||
      (params.phase == 0u &&
       (params.count == 0u || control.arena.buffer == VK_NULL_HANDLE ||
        !DecodePipelineNestedPhase(params.failed_nested_phase,
                                   failed_phase)))) {
    return false;
  }
  params.generation_stride = control.generation_stride;
  params.declared_step_count = control.declared_step_count;
  std::array<VkBufferMemoryBarrier, 2u> inputs{};
  std::uint32_t input_count = 0u;
  if (control.arena.buffer != VK_NULL_HANDLE) {
    inputs[input_count++] = VulkanBufferBarrier(
        control.arena, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
  }
  inputs[input_count++] = VulkanBufferBarrier(
      control.summary, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
  vkCmdPipelineBarrier(command,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_HOST_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                       input_count, inputs.data(), 0u, nullptr);
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     control.reduce_pipeline->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        control.reduce_pipeline->pipeline_layout, 0u, 1u,
                        &control.reduce_descriptor, 0u, nullptr);
  PushVulkanConstants(command, control.reduce_pipeline->pipeline_layout,
                      VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(params), &params);
  DispatchVulkan(command, 1u, 1u, 1u);
  return true;
}

} // namespace

bool FoldVulkanPipelineControl(
    const VkCommandBuffer command,
    const VulkanPipelineControlResources &control,
    const PreparedProgramStatusSlice slice, const std::uint32_t declared_step,
    const std::uint32_t failed_outer_window,
    const std::uint32_t failed_inner_iteration,
    const std::uint32_t failed_nested_phase) noexcept {
  return EncodeVulkanPipelineControl(
      command, control,
      VulkanPipelineControlParams{.first = slice.first,
                                  .count = slice.count,
                                  .declared_step = declared_step,
                                  .failed_outer_window = failed_outer_window,
                                  .failed_inner_iteration =
                                      failed_inner_iteration,
                                  .failed_nested_phase = failed_nested_phase});
}

bool OpenVulkanPipelineControl(
    const VkCommandBuffer command,
    const VulkanPipelineControlResources &control) noexcept {
  return EncodeVulkanPipelineControl(command, control,
                                     VulkanPipelineControlParams{.phase = 2u});
}

bool FinishVulkanPipelineControl(
    const VkCommandBuffer command,
    const VulkanPipelineControlResources &control,
    const PreparedPipelineStatusLayout &) noexcept {
  return EncodeVulkanPipelineControl(command, control,
                                     VulkanPipelineControlParams{.phase = 1u});
}

} // namespace rund::node::accel::detail

#endif
