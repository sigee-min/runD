#pragma once

#include "state.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

void EncodeVulkanRangeControl(const VulkanRangeResources &range,
                              const VkCommandBuffer command) {
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     range.control_pipeline->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        range.control_pipeline->pipeline_layout, 0u, 1u,
                        &range.control_descriptor, 0u, nullptr);
  PushVulkanConstants(command, range.control_pipeline->pipeline_layout,
                      VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                      sizeof(range.control_push), &range.control_push);
  DispatchVulkan(command, range.stage_count, 1u, 1u);
}

void EncodeVulkanRangeStage(const VulkanRangeResources &range,
                            const VkCommandBuffer command,
                            const std::uint32_t stage_index) {
  const RangeStagePlan stage = range.range.stage(stage_index);
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     range.pipelines[stage_index]->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        range.pipelines[stage_index]->pipeline_layout, 0u, 1u,
                        &range.descriptor_sets[stage_index], 0u, nullptr);
  if (range.controlled) {
    DispatchVulkanIndirect(command, range.control_indirect.buffer,
                           static_cast<VkDeviceSize>(stage_index) *
                               sizeof(RangeIndirect));
  } else {
    DispatchVulkan(command, static_cast<std::uint32_t>(stage.groups), 1u, 1u);
  }
}

} // namespace
#endif

} // namespace rund::node::accel::detail
