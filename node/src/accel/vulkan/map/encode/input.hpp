#pragma once

#include "../local.hpp"

#include <algorithm>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] bool
BindVulkanMapInputs(VulkanAdapter &adapter, const VulkanMapEncodeResources &map,
                    const rund::kernel::ComputeDispatchWindow &window,
                    VkDescriptorBufferInfo *const infos) {
  if (map.prepared == nullptr ||
      map.prepared->input_plans.size() !=
          map.prepared->plan.input_buffer_count) {
    SetVulkanLastError(adapter, "compute_plan_invalid");
    return false;
  }
  for (rund::kernel::u64 index = 0u;
       index < map.prepared->plan.input_buffer_count;
       ++index) {
    VkDeviceSize byte_offset = 0u;
    VkDeviceSize byte_range = 0u;
    const char *reason = "ok";
    const rund::kernel::ResidentBufferRef *const ref =
        map.bindings.resident_inputs.ref(index);
    const rund::kernel::ComputeDispatchWindow span =
        InputWindow(map.prepared->input_plans[static_cast<std::size_t>(index)],
                    window);
    if (ref == nullptr ||
        !VulkanMapResidentWindowSpan(adapter, *ref, span, byte_offset,
                                     byte_range, reason)) {
      SetVulkanLastError(adapter, reason);
      return false;
    }
    const VulkanResidentBufferResult &input = map.resident.input(index);
    if (!input.check.ok || input.device_buffer == nullptr) {
      SetVulkanLastError(adapter, input.check.reason);
      return false;
    }
    infos[static_cast<std::size_t>(index + 1u)] = VkDescriptorBufferInfo{
        input.device_buffer->buffer, byte_offset, byte_range};
  }
  return true;
}

} // namespace
#endif

} // namespace rund::node::accel::detail
