#pragma once

#include <accel/check.hpp>

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

struct VulkanScatterEncodeState {
  VulkanScatterEncodeResources *scatter = nullptr;
  VkCommandBuffer command = VK_NULL_HANDLE;
  std::uint32_t workgroups = 0u;
};

[[nodiscard]] rund::AccelCheck LoadVulkanScatterEncodeState(
    VulkanAdapter &adapter, const std::shared_ptr<void> &resources,
    void *const command_buffer_raw, VulkanScatterEncodeState &state) {
  state.scatter = static_cast<VulkanScatterEncodeResources *>(resources.get());
  state.command = reinterpret_cast<VkCommandBuffer>(command_buffer_raw);
  if (state.scatter == nullptr || state.scatter->adapter != &adapter ||
      state.command == VK_NULL_HANDLE || state.scatter->pipeline == nullptr ||
      state.scatter->output == nullptr) {
    SetVulkanLastError(adapter, "compute_scatter_invalid");
    return rund::AccelCheck{false, "compute_scatter_invalid"};
  }
  state.workgroups = static_cast<std::uint32_t>(
      (state.scatter->plan.element_count + kScatterBlockSize - 1u) /
      kScatterBlockSize);
  return rund::AccelCheck{true, "ok"};
}

} // namespace
#endif

} // namespace rund::node::accel::detail
