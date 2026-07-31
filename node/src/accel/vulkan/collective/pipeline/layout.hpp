#pragma once

#include "../pipeline.hpp"
#include "storage.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] bool
CreateCollectiveDescriptorSetLayout(VulkanAdapter &adapter,
                                    const std::uint32_t descriptor_count,
                                    VkDescriptorSetLayout &layout) {
  if (descriptor_count == 0u) {
    SetVulkanLastError(adapter, "accel_vulkan_spirv_invalid");
    return false;
  }
  std::array<VkDescriptorSetLayoutBinding, 8u> inline_bindings{};
  std::vector<VkDescriptorSetLayoutBinding> heap_bindings{};
  VkDescriptorSetLayoutBinding *const bindings = CollectiveLayoutBindings(
      descriptor_count, inline_bindings, heap_bindings);
  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = descriptor_count;
  layout_info.pBindings = bindings;
  if (vkCreateDescriptorSetLayout(adapter.device, &layout_info, nullptr,
                                  &layout) != VK_SUCCESS ||
      layout == VK_NULL_HANDLE) {
    SetVulkanLastError(adapter, "accel_vulkan_descriptor_unavailable");
    return false;
  }
  return true;
}

[[nodiscard]] bool CreateCollectivePipelineLayout(
    VulkanAdapter &adapter, const VkDescriptorSetLayout set_layout,
    const std::uint32_t push_bytes, VkPipelineLayout &layout) {
  VkPushConstantRange push{};
  push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push.offset = 0u;
  push.size = push_bytes;
  VkPipelineLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layout_info.setLayoutCount = 1u;
  layout_info.pSetLayouts = &set_layout;
  layout_info.pushConstantRangeCount = push_bytes == 0u ? 0u : 1u;
  layout_info.pPushConstantRanges = push_bytes == 0u ? nullptr : &push;
  if (vkCreatePipelineLayout(adapter.device, &layout_info, nullptr, &layout) !=
          VK_SUCCESS ||
      layout == VK_NULL_HANDLE) {
    SetVulkanLastError(adapter, "accel_vulkan_pipeline_unavailable");
    return false;
  }
  return true;
}

} // namespace
#endif

} // namespace rund::node::accel::detail
