#pragma once

#include <accel/check.hpp>

#include "../local/api.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

struct VulkanSortEncodeState {
  VulkanSortEncodeResources *sort = nullptr;
  VkCommandBuffer command = VK_NULL_HANDLE;
};

[[nodiscard]] rund::AccelCheck LoadVulkanSortEncodeState(
    VulkanAdapter &adapter, const std::shared_ptr<void> &resources,
    void *const command_buffer_raw, VulkanSortEncodeState &state) {
  state.sort = static_cast<VulkanSortEncodeResources *>(resources.get());
  state.command = reinterpret_cast<VkCommandBuffer>(command_buffer_raw);
  if (state.sort == nullptr || state.sort->adapter != &adapter ||
      state.command == VK_NULL_HANDLE ||
      state.sort->dispatch_pipeline == nullptr ||
      state.sort->classify_pipeline == nullptr ||
      state.sort->prefix_pipeline == nullptr ||
      state.sort->base_pipeline == nullptr ||
      state.sort->scatter_pipeline == nullptr ||
      state.sort->adapter->max_dispatch_groups == 0u ||
      state.sort->chunk_count == 0u || state.sort->dispatch_count == 0u ||
      state.sort->classify_pipeline->push_bytes != kSortPushBytes ||
      state.sort->scatter_pipeline->push_bytes != kSortPushBytes) {
    SetVulkanLastError(adapter, "compute_plan_invalid");
    return rund::AccelCheck{false, "compute_plan_invalid"};
  }
  return rund::AccelCheck{true, "ok"};
}

} // namespace
#endif

} // namespace rund::node::accel::detail
