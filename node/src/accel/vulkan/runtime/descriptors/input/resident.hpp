#pragma once

#include "../../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] bool BindResidentInputDescriptor(
    VulkanAdapter& adapter,
    const rund::kernel::ComputeDispatchWindow& window,
    const rund::kernel::BindingSet& bindings,
    const VulkanResidentBindings* const resident_bindings,
    const rund::kernel::u64 index,
    VkDescriptorBufferInfo& info) {
  if (resident_bindings == nullptr) {
    SetVulkanLastError(adapter, "compute_binding_mismatch");
    return false;
  }
  const rund::kernel::ResidentBufferRef *const ref =
      bindings.resident_inputs.ref(index);
  VkDeviceSize byte_offset = 0u;
  VkDeviceSize byte_range = 0u;
  const char* reason = "ok";
  if (ref == nullptr ||
      !ResidentWindowSpan(*ref, window, byte_offset, byte_range, reason)) {
    SetVulkanLastError(adapter, reason);
    return false;
  }
  const VulkanResidentBufferResult& resident_input =
      resident_bindings->input(index);
  if (!resident_input.check.ok || resident_input.device_buffer == nullptr) {
    SetVulkanLastError(adapter, resident_input.check.reason);
    return false;
  }
  info = VkDescriptorBufferInfo{
      .buffer = resident_input.device_buffer->buffer,
      .offset = byte_offset,
      .range = byte_range,
  };
  return true;
}

}  // namespace
#endif

}  // namespace rund::node::accel::detail
