#include "model.hpp"

#include "../../../scatter/reduce/status.hpp"

#include "../../collective/finish.hpp"
#include "../../kernel/ops/status.hpp"

#include <rund/compute/reason.hpp>

#include <array>
#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

rund::AccelCheck
FinishVulkanScatterReduce(VulkanAdapter &adapter,
                          const std::shared_ptr<void> &resources) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const state =
      static_cast<VulkanScatterReduceResources *>(resources.get());
  const std::uint32_t *const status =
      state == nullptr ? nullptr : VulkanStatusValue(state->status);
  if (state == nullptr || state->adapter != &adapter || status == nullptr) {
    return {false, "compute_scatter_reduce_buffer_invalid"};
  }
  const rund::AccelCheck check = ScatterReduceStatus(status[0]);
  if (!check.ok) {
    return check;
  }
  return AcceptVulkanDispatches(adapter, state->plan.pass_count);
#else
  (void)adapter;
  (void)resources;
  return {false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck DescribeVulkanScatterReducePipelineStatus(
    const std::shared_ptr<void> &resources,
    VulkanPipelineStatusSource &source) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const state =
      static_cast<VulkanScatterReduceResources *>(resources.get());
  constexpr std::array mapping{
      VulkanPipelineStatusMapping{
          1u, rund::compute::Reason::ScatterReduceCountOutOfRange},
      VulkanPipelineStatusMapping{
          2u, rund::compute::Reason::ScatterReduceIndexOutOfRange}};
  return state == nullptr
             ? rund::AccelCheck{false, "compute_scatter_reduce_buffer_invalid"}
             : DescribeVulkanPipelineStatus(state->status, 1u,
                                            VulkanPipelineStatusRule::Exact, 0u,
                                            mapping, source);
#else
  (void)resources;
  (void)source;
  return {false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck DescribeVulkanScatterReducePipelineTelemetry(
    const std::shared_ptr<void> &resources,
    VulkanPipelineTelemetrySource &source) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  source = {};
  const auto *const state =
      static_cast<const VulkanScatterReduceResources *>(resources.get());
  if (state == nullptr || state->status.device.buffer == VK_NULL_HANDLE ||
      state->status.device.bytes < state->plan.status_bytes) {
    return {false, "compute_scatter_reduce_buffer_invalid"};
  }
  source = VulkanPipelineTelemetrySource{
      .kind = VulkanPipelineTelemetryKind::IndexedControl,
      .primary = &state->status.device,
      .capacity = state->plan.element_count,
      .work_item_count = state->plan.output_count,
      .primary_word_count = 4u,
      .indirect_dispatch_count = 2u,
  };
  return {true, "ok"};
#else
  (void)resources;
  (void)source;
  return {false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
