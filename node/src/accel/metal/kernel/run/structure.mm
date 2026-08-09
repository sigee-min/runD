#include "../../../context/internal/support.hpp"
#include "../../../kernel/backend/execute.hpp"
#include "../../../kernel/backend/template_plan.hpp"
#include "../../../kernel/status.hpp"

#include "../../../sort/block/metal.hpp"
#include "../../buffer/owner.hpp"
#include "../../compact/local.hpp"
#include "../../gather/local.hpp"
#include "../../histogram/local.hpp"
#include "../../kernel.hpp"
#include "../../numeric/source.hpp"
#include "../../numeric/state.hpp"
#include "../../partition/local.hpp"
#include "../../pipeline/guard.hpp"
#include "../../pipeline/source_recipe.hpp"
#include "../../range/local.hpp"
#include "../../reduce/local.hpp"
#include "../../runtime/map/source_upper.hpp"
#include "../../scan/local.hpp"
#include "../../scan/source.hpp"
#include "../../scatter/local.hpp"
#include "../../scatter/reduce/model.hpp"
#include "../../segmented/local.hpp"
#include "../../segmented/reduce/model.hpp"
#include "../../sort/source.hpp"
#include "../manifest.hpp"
#include "../ops/prepare.hpp"
#include "../pipeline/build.hpp"
#include "../pipeline/identity_index.hpp"
#include "map_memory.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] bool AddMetalViewDispatchCount(
    const KernelExecutionStep &step, const KernelViewLayout *const views,
    PreparedKernelRouteReservation &reservation) noexcept {
  if (views == nullptr || step.kind() == rund::kernel::NodeKind::Map ||
      step.kind() == rund::kernel::NodeKind::ScatterReduce) {
    return true;
  }
  std::uint64_t count = 0u;
  for (std::size_t local = 0u; local < step.graph_binding_indices.size();
       ++local) {
    const std::uint64_t binding = step.graph_binding_indices[local];
    bool first = true;
    for (std::size_t prior = 0u; prior < local; ++prior) {
      if (step.graph_binding_indices[prior] == binding) {
        first = false;
        break;
      }
    }
    if (!first) {
      continue;
    }
    if (std::any_of(views->begin(), views->end(),
                    [binding](const KernelViewSlot &view) {
                      return view.binding == binding;
                    }) &&
        !backend_template_plan::add(count, 1u)) {
      return false;
    }
  }
  if (count != 0u) {
    reservation.capture_binding_slot_upper =
        std::max(reservation.capture_binding_slot_upper, std::uint64_t{3u});
  }
  return backend_template_plan::add(reservation.dispatch_count, count);
}

[[nodiscard]] bool AddMetalPrimitiveDispatchUpper(
    const rund::kernel::ComputePlan &plan,
    const PreparedBackendManifest &manifest,
    PreparedKernelRouteReservation &reservation) noexcept {
  if (plan.dispatch_count == 0u || manifest.pipeline_stage_count == 0u) {
    return false;
  }
  std::uint64_t additional = 0u;
  return backend_template_plan::product(plan.dispatch_count,
                                        manifest.pipeline_stage_count - 1u,
                                        additional) &&
         backend_template_plan::add(reservation.dispatch_count, additional);
}

[[nodiscard]] rund::AccelCheck PlanMetalStepStructure(
    const KernelExecutionStep &step, const rund::kernel::ComputePlan &plan,
    const BoundStep *const bound, const KernelViewLayout *const views,
    const std::uint64_t, PreparedKernelRouteReservation &reservation) noexcept {
  const PreparedBackendManifest manifest =
      BuildMetalBackendManifest(step, plan, bound, 1u);
  if (!manifest.ok) {
    return rund::AccelCheck{false, manifest.reason};
  }
  reservation.capture_binding_slot_upper =
      std::max(reservation.capture_binding_slot_upper,
               manifest.capture_binding_slot_upper);
  if (!AddMetalPrimitiveDispatchUpper(plan, manifest, reservation) ||
      !AddMetalViewDispatchCount(step, views, reservation)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  std::uint64_t route = 0u;
  switch (step.kind()) {
  case rund::kernel::NodeKind::Map: {
    std::uint64_t ignored_source_bytes = 0u;
    std::uint64_t transient_bytes = 0u;
    if (!AddMetalMapTemplateHostBytes(
            reservation.template_host_bytes, plan,
            MetalMapUniqueCheckCount(step.artifact)) ||
        !AddMetalMapRouteHostBytes(route, plan) ||
        !backend_template_plan::map_source_upper(
            step, plan, ignored_source_bytes, transient_bytes)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    reservation.source_transient_bytes =
        std::max(reservation.source_transient_bytes, transient_bytes);
    break;
  }
  case rund::kernel::NodeKind::Scan:
    route = sizeof(MetalScanEncodeResources);
    break;
  case rund::kernel::NodeKind::SegmentedScan:
    route = sizeof(MetalSegmentedScanEncodeResources);
    break;
  case rund::kernel::NodeKind::SegmentedReduce:
    route = sizeof(MetalSegmentedReduceResources);
    break;
  case rund::kernel::NodeKind::Sort:
    route = sizeof(MetalSortEncodeResources);
    break;
  case rund::kernel::NodeKind::Compact:
    route = sizeof(MetalCompactEncodeResources);
    break;
  case rund::kernel::NodeKind::Gather:
    route = sizeof(MetalGatherEncodeResources);
    break;
  case rund::kernel::NodeKind::Histogram:
    route = sizeof(MetalHistogramEncodeResources);
    break;
  case rund::kernel::NodeKind::Partition:
    route = sizeof(MetalPartitionEncodeResources);
    break;
  case rund::kernel::NodeKind::Reduce:
    route = sizeof(MetalReduceEncodeResources);
    break;
  case rund::kernel::NodeKind::Scatter:
    route = sizeof(MetalScatterEncodeResources);
    break;
  case rund::kernel::NodeKind::ScatterReduce:
    route = sizeof(MetalScatterReduceResources);
    break;
  case rund::kernel::NodeKind::Stencil:
  case rund::kernel::NodeKind::Window:
    route = sizeof(MetalRangeResources);
    break;
  case rund::kernel::NodeKind::Transform:
  case rund::kernel::NodeKind::Matrix:
  case rund::kernel::NodeKind::Factor:
  case rund::kernel::NodeKind::Solve:
  case rund::kernel::NodeKind::Spectrum:
    route = sizeof(MetalNumericPrepared);
    break;
  }
  const std::uint64_t immutable_host_bytes =
      step.kind() == rund::kernel::NodeKind::Map
          ? 0u
          : sizeof(MetalKernelImmutablePipelines);
  reservation.source_transient_bytes = std::max(
      reservation.source_transient_bytes, manifest.cold_source_transient_bytes);
  return backend_template_plan::add(reservation.route_host_bytes, route) &&
                 backend_template_plan::add(reservation.template_host_bytes,
                                            immutable_host_bytes) &&
                 backend_template_plan::add(reservation.template_source_bytes,
                                            manifest.cold_cache_source_bytes) &&
                 backend_template_plan::add(
                     reservation.template_host_bytes,
                     manifest.cold_cache_source_storage_bytes) &&
                 backend_template_plan::add(
                     reservation.template_native_allocation_count,
                     manifest.cold_cache_native_object_count) &&
                 backend_template_plan::add(reservation.status_source_count,
                                            manifest.status_source_count) &&
                 backend_template_plan::add(reservation.status_entry_count,
                                            manifest.status_entry_count) &&
                 backend_template_plan::add(reservation.status_command_count,
                                            manifest.status_command_count) &&
                 backend_template_plan::add(reservation.status_parameter_bytes,
                                            manifest.status_parameter_bytes) &&
                 backend_template_plan::add(reservation.telemetry_source_count,
                                            manifest.telemetry_source_count)
             ? rund::AccelCheck{true, "ok"}
             : rund::AccelCheck{false, "compute_pipeline_capacity"};
}

[[nodiscard]] backend_template_plan::BackendShape MetalBackendShape() noexcept {
  return backend_template_plan::BackendShape{
      .storage_alignment = 1u,
      .max_dispatch_groups = 1u,
      .reset_dispatch_window = std::numeric_limits<std::uint32_t>::max(),
      .template_capacity = PreparedPipelineStepCapacity,
      .route_header_bytes = sizeof(MetalKernelResources),
      .route_step_bytes = sizeof(MetalKernelEntry),
      .route_inline_step_capacity = kInlineBoundStepCapacity,
      .template_header_bytes = sizeof(MetalKernelProgramTemplate),
      .template_step_bytes = sizeof(MetalKernelProgramStepTemplate),
      .plan_step = PlanMetalStepStructure,
  };
}

} // namespace
#endif

rund::AccelCheck PlanMetalPipelinePrivateKernel(
    const BackendRun &run,
    PreparedKernelRouteReservation &reservation) noexcept {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  return backend_template_plan::plan(run, MetalBackendShape(), reservation);
#else
  (void)run;
  reservation = {};
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

rund::AccelCheck
PlanMetalPipelineProgram(const KernelExecution &execution,
                         const PreparedKernelProgramRoute &route,
                         PreparedKernelRouteReservation &reservation) noexcept {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  return backend_template_plan::plan_program(execution, route,
                                             MetalBackendShape(), reservation);
#else
  (void)execution;
  (void)route;
  reservation = {};
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
