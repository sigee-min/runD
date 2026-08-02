#include <accel/check.hpp>

#include "../kernel/ops/status.hpp"
#include "local/api.hpp"
#include <rund/counter.hpp>

namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelCheck
DescribeVulkanSortPipelineStatus(const std::shared_ptr<void> &resources,
                                 VulkanPipelineStatusSource &source) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const sort = static_cast<VulkanSortEncodeResources *>(resources.get());
  constexpr std::array mapping{
      VulkanPipelineStatusMapping{std::numeric_limits<std::uint32_t>::max(),
                                  rund::compute::Reason::BoundedCountInvalid}};
  return sort == nullptr
             ? rund::AccelCheck{false, "compute_plan_invalid"}
             : DescribeVulkanPipelineStatus(sort->status, 1u,
                                            VulkanPipelineStatusRule::BitFlags,
                                            0u, mapping, source);
#else
  (void)resources;
  (void)source;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

[[nodiscard]] rund::AccelCheck
DescribeVulkanSortPipelineTelemetry(const std::shared_ptr<void> &resources,
                                    VulkanPipelineTelemetrySource &source) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  source = {};
  const auto *const sort =
      static_cast<const VulkanSortEncodeResources *>(resources.get());
  if (sort == nullptr) {
    return {false, "compute_plan_invalid"};
  }
  if (sort->control.iteration == 0u) {
    return {true, "ok"};
  }
  if (!sort->control.has_count() || sort->logical_count.buffer == nullptr ||
      sort->plan.radix_pass_count >
          std::numeric_limits<std::uint32_t>::max() / 2u) {
    return {false, "compute_plan_invalid"};
  }
  source = VulkanPipelineTelemetrySource{
      .kind = VulkanPipelineTelemetryKind::ControlledCollective,
      .primary = sort->logical_count.buffer,
      .count = sort->logical_count.buffer,
      .control = sort->control,
      .count_offset =
          sort->logical_count.offset + sort->control.count_byte_offset,
      .capacity = sort->control.capacity,
      .indirect_dispatch_count = sort->plan.radix_pass_count * 2u,
  };
  return {true, "ok"};
#else
  (void)resources;
  (void)source;
  return {false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck DescribeVulkanSortPipelineCaptureDemand(
    const std::shared_ptr<void> &resources,
    std::uint64_t &indirect_dispatch_count) noexcept {
  indirect_dispatch_count = 0u;
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const auto *const sort =
      static_cast<const VulkanSortEncodeResources *>(resources.get());
  constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  if (sort == nullptr || sort->pass_count == 0u || sort->chunk_count == 0u ||
      sort->pass_count > maximum / 2u ||
      static_cast<std::uint64_t>(sort->chunk_count) >
          maximum / (2u * static_cast<std::uint64_t>(sort->pass_count))) {
    return {false, "compute_plan_invalid"};
  }
  indirect_dispatch_count =
      2u * static_cast<std::uint64_t>(sort->pass_count) * sort->chunk_count;
  return {true, "ok"};
#else
  (void)resources;
  return {false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck FinishVulkanSort(VulkanAdapter &adapter,
                                  const std::shared_ptr<void> &resources) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const sort = static_cast<VulkanSortEncodeResources *>(resources.get());
  if (sort == nullptr || sort->adapter != &adapter) {
    SetVulkanLastError(adapter, "compute_plan_invalid");
    return rund::AccelCheck{false, "compute_plan_invalid"};
  }
  const auto *const status = VulkanStatusValue(sort->status);
  if (status == nullptr) {
    SetVulkanLastError(adapter, "accel_vulkan_memory_unavailable");
    return rund::AccelCheck{false, "accel_vulkan_memory_unavailable"};
  }
  const rund::kernel::u32 flags = status[0];
  ::rund::detail::counter::Accumulate(adapter.dispatch_count,
                                      sort->dispatch_count);
  if (flags != 0u) {
    const char *const reason = "compute_bounded_count_invalid";
    SetVulkanLastError(adapter, reason);
    return rund::AccelCheck{false, reason};
  }
  SetVulkanLastError(adapter, "ok");
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
