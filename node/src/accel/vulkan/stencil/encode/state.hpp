#pragma once

#include <accel/check.hpp>

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

struct VulkanStencilEncodeState {
  VulkanStencilEncodeResources *stencil = nullptr;
  VkCommandBuffer command = VK_NULL_HANDLE;
  std::uint32_t workgroups = 0u;
};

[[nodiscard]] rund::AccelCheck LoadVulkanStencilEncodeState(
    VulkanAdapter &adapter, const std::shared_ptr<void> &resources,
    void *const command_buffer_raw, VulkanStencilEncodeState &state) {
  state.stencil = static_cast<VulkanStencilEncodeResources *>(resources.get());
  state.command = reinterpret_cast<VkCommandBuffer>(command_buffer_raw);
  if (state.stencil == nullptr || state.stencil->adapter != &adapter ||
      state.command == VK_NULL_HANDLE || state.stencil->pipeline == nullptr ||
      state.stencil->input == nullptr || state.stencil->output == nullptr) {
    SetVulkanLastError(adapter, "compute_stencil_invalid");
    return rund::AccelCheck{false, "compute_stencil_invalid"};
  }
  state.workgroups = static_cast<std::uint32_t>(
      (state.stencil->plan.element_count + kStencilBlockSize - 1u) /
      kStencilBlockSize);
  return rund::AccelCheck{true, "ok"};
}

} // namespace
#endif

} // namespace rund::node::accel::detail
