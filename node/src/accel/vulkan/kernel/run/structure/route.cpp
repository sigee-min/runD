#include "../../../adapter/access.hpp"

#include "../../../../kernel/backend/execute.hpp"
#include "../../../../kernel/backend/template/arithmetic.hpp"
#include "../../../../kernel/backend/template/model.hpp"
#include "../../../../kernel/backend/template/reservation.hpp"
#include "../../../../kernel/recurrence/plan.hpp"
#include "../../../../kernel/status.hpp"
#include "../../../../resident/window/admission/runtime/windows.hpp"

#include "../../../collective/chunk.hpp"
#include "../../../compact/local.hpp"
#include "../../../descriptor.hpp"
#include "../../../gather/local.hpp"
#include "../../../histogram/local.hpp"
#include "../../../kernel.hpp"
#include "../../../map/api.hpp"
#include "../../../map/local.hpp"
#include "../../../map/source/upper.hpp"
#include "../../../numeric/source.hpp"
#include "../../../numeric/state.hpp"
#include "../../../partition/local.hpp"
#include "../../../range/local.hpp"
#include "../../../reduce/local.hpp"
#include "../../../scan/local.hpp"
#include "../../../scan/source.hpp"
#include "../../../scatter/local.hpp"
#include "../../../scatter/reduce/model.hpp"
#include "../../../segmented/local.hpp"
#include "../../../segmented/reduce/model.hpp"
#include "../../../sort/local/state.hpp"
#include "../../manifest.hpp"
#include "../../ops/prepare.hpp"
#include "../../pipeline/capacity.hpp"
#include "../../pipeline/evidence.hpp"
#include "../../pipeline/recurrence.hpp"
#include "../../pipeline/source.hpp"
#include "../../pipeline/state.hpp"
#include "../../reset/source.hpp"

#include "../../../../primitive/block.hpp"
#include "../../../../sort/block/vulkan.hpp"

#include <kernel/program/compute/scan/plan.hpp>

#include "../route.hpp"
#include "../storage.hpp"
#include <limits>

#include "route/local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] backend_template_plan::BackendShape
VulkanBackendShape(const std::uint64_t alignment,
                   const std::uint64_t max_dispatch_groups) noexcept {
  return backend_template_plan::BackendShape{
      .storage_alignment = alignment,
      .max_dispatch_groups = max_dispatch_groups,
      .reset_dispatch_window = max_dispatch_groups * 256u,
      .template_capacity = PreparedPipelineStepCapacity,
      .route_header_bytes = sizeof(VulkanKernelResources),
      .route_step_bytes = sizeof(VulkanKernelEntry),
      .route_inline_step_capacity = kInlineBoundStepCapacity,
      .template_header_bytes = sizeof(VulkanKernelProgramTemplate),
      .template_step_bytes = sizeof(VulkanKernelProgramStepTemplate),
      .template_step_capacity = kVulkanPipelineTemplateStepCapacity,
      .plan_step = PlanVulkanStepStructure,
  };
}

} // namespace
#endif

rund::AccelCheck PlanVulkanPipelinePrivateKernel(
    const BackendRun &run,
    PreparedKernelRouteReservation &reservation) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanAdapter *const adapter =
      run.pick == nullptr ? nullptr : CheckedVulkanAdapter(*run.pick);
  const std::uint64_t alignment =
      adapter == nullptr ? 0u : run.pick->caps.storage_alignment;
  const rund::AccelCheck planned = backend_template_plan::plan(
      run,
      VulkanBackendShape(
          alignment, adapter == nullptr ? 0u : adapter->max_dispatch_groups),
      reservation);
  return !planned.ok || adapter == nullptr
             ? planned
             : CompleteVulkanRouteCaptureStructure(
                   run.views, run.resets == nullptr ? 0u : run.resets->size(),
                   *adapter, reservation);
#else
  (void)run;
  reservation = {};
  return rund::AccelCheck{false, "accel_vulkan_unavailable"};
#endif
}

rund::AccelCheck PlanVulkanPipelineProgram(
    const KernelExecution &execution, const PreparedKernelProgramRoute &route,
    PreparedKernelRouteReservation &reservation) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const rund::AccelDevice *const pick =
      execution.context_admission.pick == nullptr
          ? nullptr
          : &execution.context_admission.pick->raw;
  VulkanAdapter *const adapter =
      pick == nullptr ? nullptr : CheckedVulkanAdapter(*pick);
  const rund::AccelCheck planned = backend_template_plan::plan_program(
      execution, route,
      VulkanBackendShape(execution.admission.frozen_caps.storage_alignment,
                         adapter == nullptr ? 0u
                                            : adapter->max_dispatch_groups),
      reservation);
  return !planned.ok || adapter == nullptr
             ? planned
             : CompleteVulkanRouteCaptureStructure(
                   route.views, execution.resets.size(), *adapter, reservation);
#else
  (void)execution;
  (void)route;
  reservation = {};
  return rund::AccelCheck{false, "accel_vulkan_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
