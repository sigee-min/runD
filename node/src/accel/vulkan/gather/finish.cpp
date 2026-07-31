#include <accel/check.hpp>

#include "../../gather/status.hpp"
#include "../collective/finish.hpp"
#include "../kernel/ops/status.hpp"
#include "local.hpp"

namespace rund::node::accel::detail {

bool ObserveVulkanGatherFailure(const std::shared_ptr<void> &resources,
                                std::uint64_t &ordinal) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const auto *const gather =
      static_cast<const VulkanGatherEncodeResources *>(resources.get());
  const std::uint32_t *const status =
      gather == nullptr ? nullptr : VulkanStatusValue(gather->status);
  if (status == nullptr ||
      gather->status.read_bytes < 2u * sizeof(*status) || status[0] != 2u) {
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

[[nodiscard]] rund::AccelCheck
DescribeVulkanGatherPipelineStatus(const std::shared_ptr<void> &resources,
                                   VulkanPipelineStatusSource &source) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const gather =
      static_cast<VulkanGatherEncodeResources *>(resources.get());
  constexpr std::array mapping{
      VulkanPipelineStatusMapping{1u,
                                  rund::compute::Reason::BoundedCountInvalid},
      VulkanPipelineStatusMapping{
          2u, rund::compute::Reason::GatherIndexOutOfRange}};
  return gather == nullptr
             ? rund::AccelCheck{false, "compute_gather_invalid"}
             : DescribeVulkanPipelineStatus(gather->status, 1u,
                                            VulkanPipelineStatusRule::Exact, 0u,
                                            mapping, source);
#else
  (void)resources;
  (void)source;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

[[nodiscard]] rund::AccelCheck
DescribeVulkanGatherPipelineTelemetry(const std::shared_ptr<void> &resources,
                                      VulkanPipelineTelemetrySource &source) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  source = {};
  const auto *const gather =
      static_cast<const VulkanGatherEncodeResources *>(resources.get());
  if (gather == nullptr || gather->status.device.buffer == VK_NULL_HANDLE ||
      gather->status.device.bytes < 2u * sizeof(std::uint32_t)) {
    return {false, "compute_gather_invalid"};
  }
  source = VulkanPipelineTelemetrySource{
      .kind = VulkanPipelineTelemetryKind::GatherControl,
      .primary = &gather->status.device,
      .capacity = gather->plan.element_count,
      .primary_word_count = 2u,
      .indirect_dispatch_count = 1u,
  };
  return {true, "ok"};
#else
  (void)resources;
  (void)source;
  return {false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck FinishVulkanGather(VulkanAdapter &adapter,
                                    const std::shared_ptr<void> &resources) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanGatherEncodeResources *gather = nullptr;
  rund::AccelCheck check = LoadVulkanFinishResources(
      adapter, resources, "compute_gather_invalid", gather);
  if (!check.ok) {
    return check;
  }
  const rund::kernel::u32 *status = nullptr;
  check = ReadVulkanStatusU32(adapter, gather->status, status);
  if (!check.ok) {
    return check;
  }
  check = GatherStatus(status[0]);
  if (!check.ok) {
    SetVulkanLastError(adapter, check.reason);
    return check;
  }
  return AcceptVulkanDispatches(adapter, gather->plan.pass_count);
#else
  (void)adapter;
  (void)resources;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
