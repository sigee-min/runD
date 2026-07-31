#pragma once

#include "../../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] bool BindStagedInputDescriptor(
    VulkanAdapter &adapter, const rund::kernel::ComputeDispatchWindow &window,
    const rund::kernel::BindingSet &bindings,
    const std::span<const InputWindowPlan> input_plans,
    const VulkanWindowBuffers &buffers, const rund::kernel::u64 index,
    rund::kernel::u64 &input_cursor, VkDescriptorBufferInfo &info) {
  if (buffers.input.buffer.buffer == VK_NULL_HANDLE) {
    SetVulkanLastError(adapter, "accel_vulkan_buffer_unavailable");
    return false;
  }
  rund::kernel::u64 input_offset = 0u;
  rund::kernel::u64 input_range = 0u;
  rund::kernel::u64 next_cursor = 0u;
  if (index >= input_plans.size()) {
    SetVulkanLastError(adapter, "compute_binding_mismatch");
    return false;
  }
  const rund::kernel::ComputeDispatchWindow input_window =
      InputWindow(input_plans[static_cast<std::size_t>(index)], window);
  if (!StagedInputRange(bindings.input_buffers[index], input_window,
                        input_cursor,
                        static_cast<rund::kernel::u64>(adapter.storage_align),
                        input_offset, input_range, next_cursor)) {
    SetVulkanLastError(adapter, "compute_binding_input_stride_invalid");
    return false;
  }
  info = VkDescriptorBufferInfo{
      .buffer = buffers.input.buffer.buffer,
      .offset = static_cast<VkDeviceSize>(input_offset),
      .range = static_cast<VkDeviceSize>(input_range),
  };
  input_cursor = next_cursor;
  return true;
}

} // namespace
#endif

} // namespace rund::node::accel::detail
