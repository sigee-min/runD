#pragma once

#include "../pipeline.hpp"

#include <kernel/program/compute/lowering/vulkan/shape.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] bool CreateVulkanPipelineLayout(VulkanAdapter &adapter,
                                              VulkanCachedPipeline &pipeline) {
  VkPipelineLayoutCreateInfo pipeline_layout_info{};
  pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_info.setLayoutCount = 1u;
  pipeline_layout_info.pSetLayouts = &pipeline.descriptor_set_layout;
  const VkPushConstantRange push{
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0u,
      .size = rund::kernel::compute_lowering_detail::kVulkanMapPushBytes,
  };
  pipeline_layout_info.pushConstantRangeCount = 1u;
  pipeline_layout_info.pPushConstantRanges = &push;
  if (vkCreatePipelineLayout(adapter.device, &pipeline_layout_info, nullptr,
                             &pipeline.pipeline_layout) != VK_SUCCESS ||
      pipeline.pipeline_layout == VK_NULL_HANDLE) {
    SetVulkanLastError(adapter, "accel_vulkan_pipeline_unavailable");
    return false;
  }
  return true;
}

[[nodiscard]] bool CreateVulkanComputePipeline(VulkanAdapter &adapter,
                                               const VkShaderModule module,
                                               VulkanCachedPipeline &pipeline) {
  VkPipelineShaderStageCreateInfo stage_info{};
  stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage_info.module = module;
  stage_info.pName = "main";
  VkComputePipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_info.stage = stage_info;
  pipeline_info.layout = pipeline.pipeline_layout;
  if (vkCreateComputePipelines(adapter.device, VK_NULL_HANDLE, 1u,
                               &pipeline_info, nullptr,
                               &pipeline.pipeline) != VK_SUCCESS ||
      pipeline.pipeline == VK_NULL_HANDLE) {
    SetVulkanLastError(adapter, "accel_vulkan_pipeline_unavailable");
    return false;
  }
  return true;
}

} // namespace
#endif

} // namespace rund::node::accel::detail
