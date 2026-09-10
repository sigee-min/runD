#pragma once

#include "../../adapter/error.hpp"

#include <accel/check.hpp>

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

struct VulkanRangeState {
  VulkanRangeResources *range = nullptr;
  VkCommandBuffer command = VK_NULL_HANDLE;
};

[[nodiscard]] rund::AccelCheck
LoadVulkanRangeState(VulkanAdapter &adapter,
                     const std::shared_ptr<void> &resources,
                     void *const command_buffer_raw, VulkanRangeState &state) {
  state.range = static_cast<VulkanRangeResources *>(resources.get());
  state.command = reinterpret_cast<VkCommandBuffer>(command_buffer_raw);
  if (state.range == nullptr || state.range->adapter != &adapter ||
      state.command == VK_NULL_HANDLE || state.range->stage_count == 0u ||
      state.range->stage_count != state.range->range.stage_count() ||
      state.range->input == nullptr || state.range->output == nullptr) {
    SetVulkanLastError(adapter, "compute_range_aggregate_invalid");
    return rund::AccelCheck{false, "compute_range_aggregate_invalid"};
  }
  const std::optional<RangeExec> execution =
      RangeExec::from(state.range->range);
  if (!execution.has_value()) {
    SetVulkanLastError(adapter, "compute_range_aggregate_invalid");
    return rund::AccelCheck{false, "compute_range_aggregate_invalid"};
  }
  if (state.range->controlled !=
          state.range->range.shape().resident_counted() ||
      (state.range->controlled &&
       (state.range->control_pipeline == nullptr ||
        state.range->control_descriptor == VK_NULL_HANDLE ||
        state.range->control_params.buffer == VK_NULL_HANDLE ||
        state.range->control_indirect.buffer == VK_NULL_HANDLE ||
        state.range->control_status.device.buffer == VK_NULL_HANDLE ||
        state.range->control_param_stride < sizeof(RangeParams)))) {
    SetVulkanLastError(adapter, "compute_range_aggregate_invalid");
    return rund::AccelCheck{false, "compute_range_aggregate_invalid"};
  }
  for (std::size_t index = 0u; index < state.range->stage_count; ++index) {
    if (state.range->pipelines[index] == nullptr ||
        (!state.range->controlled &&
         state.range->params[index].buffer == VK_NULL_HANDLE) ||
        state.range->descriptor_sets[index] == VK_NULL_HANDLE ||
        !execution->stage_dispatch_fits(index, adapter.max_dispatch_groups)) {
      SetVulkanLastError(adapter, "compute_dispatch_overflow");
      return rund::AccelCheck{false, "compute_dispatch_overflow"};
    }
  }
  return rund::AccelCheck{true, "ok"};
}

} // namespace
#endif

} // namespace rund::node::accel::detail
