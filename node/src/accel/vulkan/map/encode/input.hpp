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
  for (rund::kernel::u64 index = 0u; index < map.plan.input_buffer_count;
       ++index) {
    VkDeviceSize byte_offset = 0u;
    VkDeviceSize byte_range = 0u;
    const char *reason = "ok";
    const rund::kernel::ResidentBufferRef *const ref =
        map.bindings.resident_inputs.ref(index);
    const auto route =
        std::find_if(map.read_routes.begin(), map.read_routes.end(),
                     [&](const rund::kernel::ReadRoute value) {
                       return value.source == index;
                     });
    const rund::kernel::ComputeDispatchWindow span =
        route == map.read_routes.end()
            ? window
            : rund::kernel::ComputeDispatchWindow{
                  .begin_sequence = 0u, .tile_count = route->count};
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
