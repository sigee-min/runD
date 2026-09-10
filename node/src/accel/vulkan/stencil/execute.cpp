#include "../adapter/error.hpp"
#include "../adapter/access.hpp"

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../stencil/bindings.hpp"
#include "../../stencil/range.hpp"
#include "../../stencil/vulkan.hpp"
#include "../collective/execute.hpp"
#include "../range/local.hpp"
#include "../range/resources/lookup.hpp"

#include <string_view>

namespace rund::node::accel::detail {

rund::AccelCheck PrepareVulkanStencil(
    const rund::AccelDevice &pick, const rund::kernel::StencilDesc &desc,
    const rund::kernel::StencilPlan &plan,
    const rund::kernel::ComputeDomain domain, const RangeBinds &bindings,
    const RangePlan &range, std::shared_ptr<void> &resources,
    const VulkanKernelImmutablePipelines *const pipelines) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  resources.reset();
  VulkanAdapter *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr) {
    return rund::AccelCheck{false, "accel_vulkan_unavailable"};
  }
  if (!StencilBindingsMatch(desc, plan, bindings) ||
      !StencilRangeMatches(plan, domain, range)) {
    SetVulkanLastError(*adapter, "compute_stencil_invalid");
    return rund::AccelCheck{false, "compute_stencil_invalid"};
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
    SetVulkanLastError(*adapter, "compute_stencil_invalid");
    return rund::AccelCheck{false, "compute_stencil_invalid"};
  }
  const rund::AccelCheck check =
      PrepareVulkanRange(pick, range, *range_bindings,
                         rund::kernel::NodeKind::Stencil, resources, pipelines);
  if (!check.ok &&
      std::string_view{check.reason} == "compute_range_aggregate_invalid") {
    SetVulkanLastError(*adapter, "compute_stencil_invalid");
    return rund::AccelCheck{false, "compute_stencil_invalid"};
  }
  return check;
#else
  (void)pick;
  (void)desc;
  (void)plan;
  (void)domain;
  (void)bindings;
  (void)range;
  (void)resources;
  (void)pipelines;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck EncodeVulkanStencil(VulkanAdapter &adapter,
                                     const std::shared_ptr<void> &resources,
                                     void *const command_buffer_raw) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const rund::AccelCheck check =
      EncodeVulkanRange(adapter, resources, command_buffer_raw);
  if (!check.ok &&
      std::string_view{check.reason} == "compute_range_aggregate_invalid") {
    SetVulkanLastError(adapter, "compute_stencil_invalid");
    return rund::AccelCheck{false, "compute_stencil_invalid"};
  }
  return check;
#else
  (void)adapter;
  (void)resources;
  (void)command_buffer_raw;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck ExecuteVulkanStencil(const rund::AccelDevice &pick,
                                      const rund::kernel::StencilDesc &desc,
                                      const rund::kernel::StencilPlan &plan,
                                      const rund::kernel::ComputeDomain domain,
                                      const RangeBinds &bindings,
                                      const RangePlan &range) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  return ExecuteVulkanDomainCollective(
      pick, desc, plan, domain, bindings,
      [&range](const rund::AccelDevice &device,
               const rund::kernel::StencilDesc &operation,
               const rund::kernel::StencilPlan &prepared,
               const rund::kernel::ComputeDomain active_domain,
               const RangeBinds &resident, std::shared_ptr<void> &resources) {
        return PrepareVulkanStencil(device, operation, prepared, active_domain,
                                    resident, range, resources, nullptr);
      },
      EncodeVulkanStencil, FinishVulkanStencil);
#else
  (void)domain;
  (void)range;
  return RejectVulkanCollectiveExecute(pick, desc, plan, bindings);
#endif
}

} // namespace rund::node::accel::detail
