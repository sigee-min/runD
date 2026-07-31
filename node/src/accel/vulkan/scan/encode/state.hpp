#pragma once

#include <accel/check.hpp>

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

struct VulkanScanEncodeState {
  VulkanScanEncodeResources *scan = nullptr;
  VkCommandBuffer command = VK_NULL_HANDLE;
};

[[nodiscard]] rund::AccelCheck LoadVulkanScanEncodeState(
    VulkanAdapter &adapter, const std::shared_ptr<void> &resources,
    void *const command_buffer_handle, VulkanScanEncodeState &state) {
  state.scan = static_cast<VulkanScanEncodeResources *>(resources.get());
  state.command = reinterpret_cast<VkCommandBuffer>(command_buffer_handle);
  if (state.scan == nullptr || state.scan->adapter != &adapter ||
      state.command == VK_NULL_HANDLE || state.scan->block == nullptr ||
      state.scan->block->pipeline == VK_NULL_HANDLE ||
      state.scan->block->push_bytes != kScanPushBytes ||
      state.scan->block_set == VK_NULL_HANDLE ||
      state.scan->status == nullptr ||
      state.scan->block_count == 0u ||
      state.scan->adapter->max_dispatch_groups == 0u ||
      (state.scan->pass_count != 1u && state.scan->pass_count != 2u) ||
      (state.scan->pass_count == 2u &&
       (state.scan->prefix == nullptr || state.scan->offset == nullptr ||
        state.scan->prefix->pipeline == VK_NULL_HANDLE ||
        state.scan->offset->pipeline == VK_NULL_HANDLE ||
        state.scan->prefix->push_bytes != 0u ||
        state.scan->offset->push_bytes != kScanPushBytes ||
        state.scan->prefix_set == VK_NULL_HANDLE ||
        state.scan->offset_set == VK_NULL_HANDLE))) {
    SetVulkanLastError(adapter, "compute_scan_invalid");
    return rund::AccelCheck{false, "compute_scan_invalid"};
  }
  return rund::AccelCheck{true, "ok"};
}

} // namespace
#endif

} // namespace rund::node::accel::detail
