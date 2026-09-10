#include "../../adapter/error.hpp"

#include <accel/check.hpp>

#include "../../collective/finish.hpp"
#include "../../kernel/ops/status.hpp"
#include "../../status.hpp"
#include "../api.hpp"
#include "../local.hpp"

#include <rund/counter.hpp>

#include <array>
#include <limits>

namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelCheck
DescribeVulkanMapPipelineStatus(const std::shared_ptr<void> &resources,
                                VulkanPipelineStatusSource &source) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const map = static_cast<VulkanMapEncodeResources *>(resources.get());
  if (map == nullptr) {
    return rund::AccelCheck{false, "compute_plan_invalid"};
  }
  if (!map->controlled()) {
    source = {};
    return rund::AccelCheck{true, "ok"};
  }
  constexpr std::array mapping{
      VulkanPipelineStatusMapping{1u, rund::compute::Reason::WorksetOverflow},
      VulkanPipelineStatusMapping{
          2u, rund::compute::Reason::GatherIndexOutOfRange}};
  return DescribeVulkanPipelineStatus(map->control_status, 1u,
                                      VulkanPipelineStatusRule::Exact, 0u,
                                      mapping, source);
#else
  (void)resources;
  (void)source;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

bool ObserveVulkanMapFailure(const std::shared_ptr<void> &resources,
                             std::uint64_t &ordinal) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const auto *const map =
      static_cast<const VulkanMapEncodeResources *>(resources.get());
  const std::uint32_t *const status =
      map == nullptr || map->prepared == nullptr ||
              map->prepared->checks.empty()
          ? nullptr
          : VulkanStatusValue(map->control_status);
  if (status == nullptr ||
      map->control_status.read_bytes < 2u * sizeof(*status) ||
      status[0] != 2u) {
    return false;
  }
  ordinal = status[1];
  return true;
#else
  (void)resources;
  (void)ordinal;
  return false;
#endif
}

rund::AccelCheck FinishVulkanMap(VulkanAdapter &adapter,
                                 const std::shared_ptr<void> &resources) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const map = static_cast<VulkanMapEncodeResources *>(resources.get());
  if (map == nullptr || map->adapter != &adapter) {
    SetVulkanLastError(adapter, "compute_plan_invalid");
    return rund::AccelCheck{false, "compute_plan_invalid"};
  }
  if (map->controlled()) {
    const rund::kernel::u32 *status = nullptr;
    const rund::AccelCheck read =
        ReadVulkanStatusU32(adapter, map->control_status, status);
    if (!read.ok) {
      return read;
    }
    if (status[0] != 0u) {
      const char *const reason = status[0] == 2u
                                     ? "compute_gather_index_out_of_range"
                                     : "compute_workset_overflow";
      SetVulkanLastError(adapter, reason);
      return rund::AccelCheck{false, reason};
    }
  }
  ::rund::detail::counter::Accumulate(
      adapter.dispatch_count,
      static_cast<rund::kernel::u64>(map->windows.size()));
  SetVulkanLastError(adapter, "ok");
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck
DescribeVulkanMapPipelineTelemetry(const std::shared_ptr<void> &resources,
                                   VulkanPipelineTelemetrySource &source) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  source = {};
  const auto *const map =
      static_cast<const VulkanMapEncodeResources *>(resources.get());
  if (map == nullptr || map->prepared == nullptr) {
    return {false, "compute_plan_invalid"};
  }
  if (!map->controlled()) {
    return {true, "ok"};
  }
  if (map->control_args.buffer.buffer == VK_NULL_HANDLE ||
      map->windows.empty() ||
      map->windows.size() > std::numeric_limits<std::uint32_t>::max() / 4u) {
    return {false, "compute_plan_invalid"};
  }
  source = VulkanPipelineTelemetrySource{
      .kind = map->prepared->checks.empty()
                  ? VulkanPipelineTelemetryKind::ControlledMap
                  : VulkanPipelineTelemetryKind::GatherControl,
      .primary = map->prepared->checks.empty() ? &map->control_args.buffer
                                               : &map->control_status.device,
      .count =
          map->control.has_count() ? map->control_count.device_buffer : nullptr,
      .predicate = map->control.has_predicate()
                       ? map->control_predicate.device_buffer
                       : nullptr,
      .control = map->control,
      .count_offset =
          map->control_count.ref.offset_bytes + map->control.count_byte_offset,
      .predicate_offset = map->control_predicate.ref.offset_bytes +
                          map->control.predicate_byte_offset,
      .capacity = map->control.capacity,
      .work_item_count =
          map->windows.back().begin_sequence + map->windows.back().tile_count,
      .primary_word_count =
          map->prepared->checks.empty()
              ? static_cast<std::uint32_t>(map->windows.size() * 4u)
              : 2u,
      .indirect_dispatch_count = static_cast<std::uint32_t>(
          1u + (!map->prepared->checks.empty() ? 1u : 0u)),
  };
  if ((map->control.has_count() && source.count == nullptr) ||
      (map->control.has_predicate() && source.predicate == nullptr)) {
    return {false, "compute_plan_invalid"};
  }
  return {true, "ok"};
#else
  (void)resources;
  (void)source;
  return {false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck DescribeVulkanMapPipelineCaptureDemand(
    const std::shared_ptr<void> &resources,
    std::uint64_t &indirect_dispatch_count) noexcept {
  indirect_dispatch_count = 0u;
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const auto *const map =
      static_cast<const VulkanMapEncodeResources *>(resources.get());
  if (map == nullptr || map->prepared == nullptr || map->windows.empty()) {
    return {false, "compute_plan_invalid"};
  }
  if (map->controlled()) {
    indirect_dispatch_count = map->windows.size();
  }
  return {true, "ok"};
#else
  (void)resources;
  return {false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
