#include "../adapter/error.hpp"
#include "../adapter/access.hpp"

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../window/shape.hpp"
#include "../../window/vulkan.hpp"
#include "../collective/execute.hpp"
#include "../range/local.hpp"
#include "../range/resources/lookup.hpp"

#include <string_view>

namespace rund::node::accel::detail {

rund::AccelCheck PrepareVulkanWindow(
    const rund::AccelDevice &pick, const rund::kernel::WindowDesc &desc,
    const rund::kernel::WindowPlan &plan, const RangeBinds &bindings,
    const RangePlan &range, std::shared_ptr<void> &resources,
    const VulkanKernelImmutablePipelines *const pipelines,
    const BoundControl *const control, const KernelPreparationMode mode) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  resources.reset();
  VulkanAdapter *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr) {
    return rund::AccelCheck{false, "accel_vulkan_unavailable"};
  }
  if (!WindowShapeOk(desc, plan, bindings) ||
      !WindowRangePlanMatches(plan, range)) {
    SetVulkanLastError(*adapter, "compute_window_invalid");
    return rund::AccelCheck{false, "compute_window_invalid"};
  }
  const VulkanRangeBufferLookup lookup =
      LookupVulkanRangeBuffers(pick, bindings);
  if (!VulkanRangeLookupOk(lookup)) {
    const char *const reason = VulkanRangeLookupReason(lookup);
    SetVulkanLastError(*adapter, reason);
    return rund::AccelCheck{false, reason};
  }
  const std::optional<VulkanRangeBinds> range_bindings =
      VulkanRangeBinds::make(lookup.input, lookup.output);
  if (!range_bindings.has_value()) {
    SetVulkanLastError(*adapter, "compute_window_invalid");
    return rund::AccelCheck{false, "compute_window_invalid"};
  }
  const rund::AccelCheck check = PrepareVulkanRange(
      pick, range, *range_bindings, rund::kernel::NodeKind::Window, resources,
      pipelines, control, mode);
  if (!check.ok &&
      std::string_view{check.reason} == "compute_range_aggregate_invalid") {
    SetVulkanLastError(*adapter, "compute_window_invalid");
    return rund::AccelCheck{false, "compute_window_invalid"};
  }
  return check;
#else
  (void)pick;
  (void)desc;
  (void)plan;
  (void)bindings;
  (void)range;
  (void)resources;
  (void)pipelines;
  (void)control;
  (void)mode;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck EncodeVulkanWindow(VulkanAdapter &adapter,
                                    const std::shared_ptr<void> &resources,
                                    void *const command_buffer_raw) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const rund::AccelCheck check =
      EncodeVulkanRange(adapter, resources, command_buffer_raw);
  if (!check.ok &&
      std::string_view{check.reason} == "compute_range_aggregate_invalid") {
    SetVulkanLastError(adapter, "compute_window_invalid");
    return rund::AccelCheck{false, "compute_window_invalid"};
  }
  return check;
#else
  (void)adapter;
  (void)resources;
  (void)command_buffer_raw;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
