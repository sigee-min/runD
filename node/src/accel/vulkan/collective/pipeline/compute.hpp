#pragma once

#include "../pipeline.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] bool CreateCollectiveComputePipeline(
    VulkanAdapter &adapter, const VkShaderModule module,
    const VkPipelineLayout layout, const VulkanSpecialization &specialization,
    VkPipeline &pipeline) {
  if (specialization.count > specialization.values.size()) {
    SetVulkanLastError(adapter, "accel_vulkan_pipeline_unavailable");
    return false;
  }
  std::array<VkSpecializationMapEntry, 4u> entries{};
  for (std::uint32_t index = 0u; index < specialization.count; ++index) {
    entries[index] = VkSpecializationMapEntry{
        .constantID = index,
        .offset = static_cast<std::uint32_t>(index * sizeof(std::uint32_t)),
        .size = sizeof(std::uint32_t),
    };
  }
  VkSpecializationInfo specialization_info{};
  specialization_info.mapEntryCount = specialization.count;
  specialization_info.pMapEntries = entries.data();
  specialization_info.dataSize = specialization.count * sizeof(std::uint32_t);
  specialization_info.pData = specialization.values.data();
  VkPipelineShaderStageCreateInfo stage_info{};
  stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage_info.module = module;
  stage_info.pName = "main";
  stage_info.pSpecializationInfo =
      specialization.count == 0u ? nullptr : &specialization_info;
  VkComputePipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_info.stage = stage_info;
  pipeline_info.layout = layout;
  if (vkCreateComputePipelines(adapter.device, VK_NULL_HANDLE, 1u,
                               &pipeline_info, nullptr,
                               &pipeline) != VK_SUCCESS ||
      pipeline == VK_NULL_HANDLE) {
    SetVulkanLastError(adapter, "accel_vulkan_pipeline_unavailable");
    return false;
  }
  return true;
}

} // namespace
#endif

} // namespace rund::node::accel::detail
