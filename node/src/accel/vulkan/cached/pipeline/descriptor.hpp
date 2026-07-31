#pragma once

#include "../pipeline.hpp"

#include <array>
#include <limits>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] bool VulkanDescriptorCountForPlan(
    VulkanAdapter& adapter,
    const rund::kernel::ComputePlan& plan,
    std::uint32_t& descriptor_count) noexcept {
  const rund::kernel::u64 value_count =
      plan.input_buffer_count + plan.output_buffer_count;
  if (value_count < plan.input_buffer_count || plan.output_buffer_count == 0u ||
      value_count > static_cast<rund::kernel::u64>(
                        std::numeric_limits<std::uint32_t>::max() - 1u)) {
    SetVulkanLastError(adapter, "accel_vulkan_descriptor_unavailable");
    return false;
  }
  descriptor_count = static_cast<std::uint32_t>(value_count + 1u);
  return true;
}

[[nodiscard]] bool CreateVulkanDescriptorLayout(
    VulkanAdapter& adapter,
    const std::uint32_t descriptor_count,
    VulkanCachedPipeline& pipeline) {
  std::array<VkDescriptorSetLayoutBinding,
             kVulkanCachedPipelineInlineDescriptorCount>
      layout_bindings{};
  std::vector<VkDescriptorSetLayoutBinding> layout_binding_heap{};
  VkDescriptorSetLayoutBinding* layout_binding_data = layout_bindings.data();
  if (descriptor_count > kVulkanCachedPipelineInlineDescriptorCount) {
    layout_binding_heap.resize(descriptor_count);
    layout_binding_data = layout_binding_heap.data();
  }
  for (std::uint32_t binding = 0u; binding < descriptor_count; ++binding) {
    layout_binding_data[binding] = {binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                    1u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  }
  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = descriptor_count;
  layout_info.pBindings = layout_binding_data;
  if (vkCreateDescriptorSetLayout(adapter.device, &layout_info, nullptr,
                                  &pipeline.descriptor_set_layout) !=
          VK_SUCCESS ||
      pipeline.descriptor_set_layout == VK_NULL_HANDLE) {
    SetVulkanLastError(adapter, "accel_vulkan_descriptor_unavailable");
    return false;
  }
  return true;
}

}  // namespace
#endif

}  // namespace rund::node::accel::detail
