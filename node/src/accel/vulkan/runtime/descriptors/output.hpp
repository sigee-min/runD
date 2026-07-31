#pragma once

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] bool BindResidentOutputDescriptor(
    VulkanAdapter& adapter,
    const rund::kernel::ComputeDispatchWindow& window,
    const rund::kernel::BindingSet& bindings,
    const VulkanResidentBindings* const resident_bindings,
    const rund::kernel::u64 output_index,
    VkDescriptorBufferInfo& info) {
  if (resident_bindings == nullptr ||
      output_index >= resident_bindings->outputs.size() ||
      resident_bindings->output(output_index).device_buffer == nullptr) {
    SetVulkanLastError(adapter, "compute_binding_mismatch");
    return false;
  }
  const rund::kernel::ResidentBufferRef *const output_ref =
      bindings.resident_outputs.ref(output_index);
  VkDeviceSize byte_offset = 0u;
  VkDeviceSize byte_range = 0u;
  const char* reason = "ok";
  if (output_ref == nullptr ||
      !ResidentWindowSpan(*output_ref, window, byte_offset, byte_range,
                          reason)) {
    SetVulkanLastError(adapter, reason);
    return false;
  }
  info = VkDescriptorBufferInfo{
      .buffer = resident_bindings->output(output_index).device_buffer->buffer,
      .offset = byte_offset,
      .range = byte_range,
  };
  return true;
}

[[nodiscard]] bool BindOutputDescriptor(
    VulkanAdapter& adapter,
    const rund::kernel::ComputePlan& plan,
    const rund::kernel::ComputeDispatchWindow& window,
    const rund::kernel::BindingSet& bindings,
    const VulkanResidentBindings* const resident_bindings,
    const VulkanWindowBuffers& buffers,
    VkDescriptorBufferInfo* const infos) {
  if (buffers.resident) {
    for (rund::kernel::u64 index = 0u; index < plan.output_buffer_count;
         ++index) {
      VkDescriptorBufferInfo& output_info = infos[static_cast<std::size_t>(
          plan.input_buffer_count + index + 1u)];
      if (!BindResidentOutputDescriptor(adapter, window, bindings,
                                        resident_bindings, index,
                                        output_info)) {
        return false;
      }
    }
    return true;
  }
  if (plan.output_buffer_count != 1u) {
    SetVulkanLastError(adapter, "compute_binding_mismatch");
    return false;
  }
  VkDescriptorBufferInfo& output_info =
      infos[static_cast<std::size_t>(plan.input_buffer_count + 1u)];
  output_info = VkDescriptorBufferInfo{
      .buffer = buffers.output.buffer.buffer,
      .offset = 0u,
      .range = buffers.output.used_bytes,
  };
  return true;
}

}  // namespace
#endif

}  // namespace rund::node::accel::detail
