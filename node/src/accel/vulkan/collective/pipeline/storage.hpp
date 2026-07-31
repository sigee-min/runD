#pragma once

#include "../pipeline.hpp"

#include <array>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] VkDescriptorSetLayoutBinding* CollectiveLayoutBindings(
    const std::uint32_t descriptor_count,
    std::array<VkDescriptorSetLayoutBinding, 8u>& inline_bindings,
    std::vector<VkDescriptorSetLayoutBinding>& heap_bindings) {
  VkDescriptorSetLayoutBinding* bindings = inline_bindings.data();
  if (descriptor_count > inline_bindings.size()) {
    heap_bindings.resize(descriptor_count);
    bindings = heap_bindings.data();
  }
  for (std::uint32_t index = 0u; index < descriptor_count; ++index) {
    bindings[index] = VkDescriptorSetLayoutBinding{
        index, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u,
        VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  }
  return bindings;
}

}  // namespace
#endif

}  // namespace rund::node::accel::detail
