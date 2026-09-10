#include "../adapter/error.hpp"

#include <accel/check.hpp>

#include "../collective/finish.hpp"
#include "../kernel/ops/status.hpp"
#include "local.hpp"

#include <rund/compute/reason.hpp>

#include <array>
#include <limits>
#include <string_view>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
rund::AccelCheck FinishVulkanRange(VulkanAdapter &adapter,
                                   const std::shared_ptr<void> &resources) {
  VulkanRangeResources *range = nullptr;
  const rund::AccelCheck check = LoadVulkanFinishResources(
      adapter, resources, "compute_range_aggregate_invalid", range);
  if (!check.ok) {
    return check;
  }
  if (range->controlled && !range->control_status.pipeline) {
    const rund::kernel::u32 *status = nullptr;
    const rund::AccelCheck read =
        ReadVulkanStatusU32(adapter, range->control_status, status);
    if (!read.ok) {
      return read;
    }
    if (status[0] != 0u) {
      SetVulkanLastError(adapter, "compute_workset_overflow");
      return rund::AccelCheck{false, "compute_workset_overflow"};
    }
  }
  return AcceptVulkanDispatches(
      adapter, range->range.stage_count() +
                   static_cast<std::uint64_t>(range->controlled));
}

rund::AccelCheck
DescribeVulkanRangePipelineStatus(const std::shared_ptr<void> &resources,
                                  VulkanPipelineStatusSource &source) {
  auto *const range = static_cast<VulkanRangeResources *>(resources.get());
  if (range == nullptr) {
    return rund::AccelCheck{false, "compute_window_invalid"};
  }
  if (!range->controlled) {
    source = {};
    return rund::AccelCheck{true, "ok"};
  }
  constexpr std::array mapping{
      VulkanPipelineStatusMapping{1u, rund::compute::Reason::WorksetOverflow}};
  return DescribeVulkanPipelineStatus(range->control_status, 1u,
                                      VulkanPipelineStatusRule::Exact, 0u,
                                      mapping, source);
}

rund::AccelCheck
DescribeVulkanRangePipelineTelemetry(const std::shared_ptr<void> &resources,
                                     VulkanPipelineTelemetrySource &source) {
  source = {};
  const auto *const range =
      static_cast<const VulkanRangeResources *>(resources.get());
  if (range == nullptr) {
    return rund::AccelCheck{false, "compute_window_invalid"};
  }
  if (!range->controlled) {
    return rund::AccelCheck{true, "ok"};
  }
  if (!range->control.has_count() ||
      range->control_count.device_buffer == nullptr ||
      range->control_indirect.buffer == VK_NULL_HANDLE ||
      range->stage_count == 0u ||
      range->stage_count > std::numeric_limits<std::uint32_t>::max() / 8u) {
    return rund::AccelCheck{false, "compute_window_invalid"};
  }
  source = VulkanPipelineTelemetrySource{
      .kind = VulkanPipelineTelemetryKind::ControlledRange,
      .primary = &range->control_indirect,
      .count = range->control_count.device_buffer,
      .control = range->control,
      .count_offset = range->control_count.ref.offset_bytes +
                      range->control.count_byte_offset,
      .capacity = range->control.capacity,
      .primary_word_count = range->stage_count * 8u,
      .indirect_dispatch_count = range->stage_count,
  };
  return rund::AccelCheck{true, "ok"};
}

rund::AccelCheck DescribeVulkanRangePipelineCaptureDemand(
    const std::shared_ptr<void> &resources,
    std::uint64_t &indirect_dispatch_count) noexcept {
  indirect_dispatch_count = 0u;
  const auto *const range =
      static_cast<const VulkanRangeResources *>(resources.get());
  if (range == nullptr) {
    return rund::AccelCheck{false, "compute_window_invalid"};
  }
  indirect_dispatch_count = range->controlled ? range->stage_count : 0u;
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
