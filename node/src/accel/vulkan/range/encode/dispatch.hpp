#pragma once

#include "state.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

void EncodeVulkanRangeStage(const VulkanRangeResources &range,
                            const VkCommandBuffer command,
                            const std::uint32_t stage_index) {
  const RangeStagePlan stage = range.range.stage(stage_index);
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     range.pipelines[stage_index]->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        range.pipelines[stage_index]->pipeline_layout, 0u, 1u,
                        &range.descriptor_sets[stage_index], 0u, nullptr);
  DispatchVulkan(command, static_cast<std::uint32_t>(stage.groups), 1u, 1u);
}

} // namespace
#endif

} // namespace rund::node::accel::detail
