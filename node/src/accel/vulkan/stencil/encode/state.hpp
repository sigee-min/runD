#pragma once

#include <accel/check.hpp>

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

struct VulkanStencilEncodeState {
  VulkanStencilEncodeResources *stencil = nullptr;
  VkCommandBuffer command = VK_NULL_HANDLE;
};

[[nodiscard]] rund::AccelCheck LoadVulkanStencilEncodeState(
    VulkanAdapter &adapter, const std::shared_ptr<void> &resources,
    void *const command_buffer_raw, VulkanStencilEncodeState &state) {
  state.stencil = static_cast<VulkanStencilEncodeResources *>(resources.get());
  state.command = reinterpret_cast<VkCommandBuffer>(command_buffer_raw);
  if (state.stencil == nullptr || state.stencil->adapter != &adapter ||
      state.command == VK_NULL_HANDLE || state.stencil->stage_count == 0u ||
      state.stencil->stage_count != state.stencil->range.stage_count() ||
      state.stencil->input == nullptr || state.stencil->output == nullptr ||
      !state.stencil->shape.valid()) {
    SetVulkanLastError(adapter, "compute_stencil_invalid");
    return rund::AccelCheck{false, "compute_stencil_invalid"};
  }
  for (std::size_t index = 0u; index < state.stencil->stage_count; ++index) {
    if (state.stencil->pipelines[index] == nullptr ||
        state.stencil->params[index].buffer == VK_NULL_HANDLE ||
        state.stencil->descriptor_sets[index] == VK_NULL_HANDLE ||
        !RangeAggregateStageDispatchFits(state.stencil->range, index,
                                         adapter.max_dispatch_groups)) {
      SetVulkanLastError(adapter, "compute_dispatch_overflow");
      return rund::AccelCheck{false, "compute_dispatch_overflow"};
    }
  }
  return rund::AccelCheck{true, "ok"};
}

} // namespace
#endif

} // namespace rund::node::accel::detail
