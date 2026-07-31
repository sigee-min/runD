#pragma once

#include <accel/check.hpp>

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

struct VulkanCompactEncodeState {
  VulkanCompactEncodeResources *compact = nullptr;
  VkCommandBuffer command = VK_NULL_HANDLE;
  std::uint32_t workgroups = 0u;
};

[[nodiscard]] rund::AccelCheck LoadVulkanCompactEncodeState(
    VulkanAdapter &adapter, const std::shared_ptr<void> &resources,
    void *const command_buffer_raw, VulkanCompactEncodeState &state) {
  state.compact = static_cast<VulkanCompactEncodeResources *>(resources.get());
  state.command = reinterpret_cast<VkCommandBuffer>(command_buffer_raw);
  if (state.compact == nullptr || state.compact->adapter != &adapter ||
      state.command == VK_NULL_HANDLE || state.compact->output == nullptr ||
      state.compact->classify_pipeline == nullptr ||
      state.compact->prefix_pipeline == nullptr ||
      state.compact->scatter_pipeline == nullptr ||
      state.compact->block_count == 0u) {
    SetVulkanLastError(adapter, "compute_plan_invalid");
    return rund::AccelCheck{false, "compute_plan_invalid"};
  }
  state.workgroups = state.compact->block_count;
  return rund::AccelCheck{true, "ok"};
}

} // namespace
#endif

} // namespace rund::node::accel::detail
