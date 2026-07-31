#pragma once

#include "../../descriptor.hpp"
#include "../../descriptor/storage.hpp"
#include "../../descriptor/update.hpp"
#include "../local.hpp"
#include "input.hpp"
#include "output.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

bool UpdateVulkanWindowDescriptorSet(
    VulkanAdapter &adapter, VulkanCachedPipeline &pipeline,
    const rund::kernel::ComputePlan &plan,
    const rund::kernel::ComputeDispatchWindow &window,
    const rund::kernel::BindingSet &bindings, const ScopedBuffer &param_buffer,
    const VulkanResidentBindings *const resident_bindings,
    const VulkanWindowBuffers &buffers, VkDescriptorSet &descriptor_set) {
  descriptor_set = DescriptorSetForPipeline(adapter, pipeline);
  if (descriptor_set == VK_NULL_HANDLE) {
    return false;
  }
  const std::uint32_t descriptor_count = static_cast<std::uint32_t>(
      plan.input_buffer_count + plan.output_buffer_count + 1u);
  VulkanDescriptorScratch scratch{};
  SelectVulkanDescriptorScratch(adapter, descriptor_count, scratch);

  scratch.infos[0] = VkDescriptorBufferInfo{
      .buffer = param_buffer.buffer.buffer,
      .offset = 0u,
      .range = param_buffer.used_bytes,
  };
  if (!BindInputDescriptors(adapter, plan, window, bindings, resident_bindings,
                            buffers, scratch.infos) ||
      !BindOutputDescriptor(adapter, plan, window, bindings, resident_bindings,
                            buffers, scratch.infos)) {
    return false;
  }
  return WriteVulkanStorageDescriptorSet(adapter, descriptor_set, scratch.infos,
                                         descriptor_count);
}

#endif

} // namespace rund::node::accel::detail
