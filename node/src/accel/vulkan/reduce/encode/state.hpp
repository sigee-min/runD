#pragma once

#include <accel/check.hpp>

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

struct VulkanReduceEncodeState {
  VulkanReduceEncodeResources *reduce = nullptr;
  VkCommandBuffer command = VK_NULL_HANDLE;
};

[[nodiscard]] rund::AccelCheck LoadVulkanReduceEncodeState(
    VulkanAdapter &adapter, const std::shared_ptr<void> &resources,
    void *const command_buffer_raw, VulkanReduceEncodeState &state) {
  state.reduce = static_cast<VulkanReduceEncodeResources *>(resources.get());
  state.command = reinterpret_cast<VkCommandBuffer>(command_buffer_raw);
  if (state.reduce == nullptr || state.reduce->adapter != &adapter ||
      state.command == VK_NULL_HANDLE || state.reduce->pipeline == nullptr ||
      state.reduce->params.buffer == VK_NULL_HANDLE ||
      state.reduce->params_stride < sizeof(ReducePassParams) ||
      state.reduce->descriptor_sets.size() !=
          static_cast<std::size_t>(state.reduce->plan.pass_count)) {
    SetVulkanLastError(adapter, "compute_reduce_invalid");
    return rund::AccelCheck{false, "compute_reduce_invalid"};
  }
  return rund::AccelCheck{true, "ok"};
}

} // namespace
#endif

} // namespace rund::node::accel::detail
