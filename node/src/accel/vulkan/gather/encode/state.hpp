#pragma once

#include <accel/check.hpp>

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

struct VulkanGatherEncodeState {
  VulkanGatherEncodeResources *gather = nullptr;
  VkCommandBuffer command = VK_NULL_HANDLE;
};

[[nodiscard]] rund::AccelCheck LoadVulkanGatherEncodeState(
    VulkanAdapter &adapter, const std::shared_ptr<void> &resources,
    void *const command_buffer_raw, VulkanGatherEncodeState &state) {
  state.gather = static_cast<VulkanGatherEncodeResources *>(resources.get());
  state.command = reinterpret_cast<VkCommandBuffer>(command_buffer_raw);
  if (state.gather == nullptr || state.gather->adapter != &adapter) {
    SetVulkanLastError(adapter, "compute_gather_invalid");
    return rund::AccelCheck{false, "compute_gather_invalid"};
  }
  if (state.command == VK_NULL_HANDLE) {
    SetVulkanLastError(adapter, "accel_vulkan_command_unavailable");
    return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
  }
  if (state.gather->control_pipeline == nullptr ||
      state.gather->gather_pipeline == nullptr) {
    SetVulkanLastError(adapter, "accel_vulkan_pipeline_unavailable");
    return rund::AccelCheck{false, "accel_vulkan_pipeline_unavailable"};
  }
  if (state.gather->output.device_buffer == nullptr ||
      state.gather->output.handle == nullptr ||
      state.gather->output.storage == nullptr) {
    SetVulkanLastError(adapter, "compute_resident_id_invalid");
    return rund::AccelCheck{false, "compute_resident_id_invalid"};
  }
  return rund::AccelCheck{true, "ok"};
}

} // namespace
#endif

} // namespace rund::node::accel::detail
