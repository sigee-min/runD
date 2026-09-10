#include "reservation_internal.hpp"

#include "../../../context/internal/execution.hpp"
#include "../../plan.hpp"
#include "../../plan/local.hpp"
#include "../../prepared/template/registry.hpp"
#include "../../reset/model.hpp"
#include "../run.hpp"

#include <kernel/core/checked.hpp>

#include <cstddef>
#include <limits>

namespace rund::node::accel::detail::backend_template_plan {

[[nodiscard]] rund::AccelCheck
plan_program(const KernelExecution &execution,
             const PreparedKernelProgramRoute &route, const BackendShape shape,
             PreparedKernelRouteReservation &reservation) noexcept {
  reservation = {};
  if (!execution.admission.check.ok || execution.steps.empty() ||
      route.kernel == nullptr || route.tile_count == 0u ||
      route.route_copies == 0u || shape.storage_alignment == 0u ||
      shape.max_dispatch_groups == 0u || shape.reset_dispatch_window == 0u ||
      shape.template_capacity == 0u || shape.route_header_bytes == 0u ||
      shape.route_step_bytes == 0u || shape.route_inline_step_capacity == 0u ||
      shape.template_header_bytes == 0u || shape.template_step_bytes == 0u ||
      shape.template_step_capacity == 0u || shape.plan_step == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  std::uint64_t route_steps = 0u;
  std::uint64_t template_steps = 0u;
  std::uint64_t view_bytes = 0u;
  std::uint64_t scratch_bytes = 0u;
  std::uint64_t reset_bytes = 0u;
  if ((execution.steps.size() > shape.route_inline_step_capacity &&
       !product(execution.steps.size(), shape.route_step_bytes, route_steps)) ||
      !product(execution.steps.size(), shape.template_step_bytes,
               template_steps) ||
      !product(route.views == nullptr ? 0u : route.views->size(),
               sizeof(KernelViewSlot), view_bytes) ||
      !product(route.scratch == nullptr ? 0u : route.scratch->size(),
               sizeof(KernelScratchPage), scratch_bytes) ||
      !product(execution.resets.size(), sizeof(BoundReset), reset_bytes)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  reservation.route_host_bytes = shape.route_header_bytes;
  reservation.template_host_bytes = shape.template_header_bytes;
  if (!add(reservation.route_host_bytes, route_steps) ||
      !add(reservation.route_host_bytes, view_bytes) ||
      !add(reservation.route_host_bytes, scratch_bytes) ||
      !add(reservation.route_host_bytes, reset_bytes) ||
      !add(reservation.template_host_bytes, template_steps)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  reservation.route_step_count = execution.steps.size();
  reservation.template_step_count = execution.steps.size();
  reservation.template_step_capacity = shape.template_step_capacity;
  reservation.template_capacity = shape.template_capacity;
  reservation.route_native_allocation_count = 1u;
  reservation.template_native_allocation_count = 1u;

  for (const ResetPlan &reset : execution.resets) {
    if (reset.binding >= execution.graph_shapes.size() ||
        !add(reservation.reset_dispatch_count,
             reset::Commands(execution.graph_shapes[reset.binding].count,
                             shape.reset_dispatch_window))) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }

  const rund::AccelRun run{.tile_count = route.tile_count};
  for (std::size_t index = 0u; index < execution.steps.size(); ++index) {
    const KernelExecutionStep &step = execution.steps[index];
    // Map planning specializes the run tile/domain metadata directly. Every
    // primitive already owns its canonical operation plan; routing it through
    // Map's ExecutionMetadata authority zeroes the primitive op hash and can
    // reject an otherwise admitted Program during public preflight.
    const rund::kernel::ComputePlan step_plan =
        step.kind() == rund::kernel::NodeKind::Map
            ? PlanStep(execution, step, run, index)
            : BuildPlannedStep(execution, step, run, index).plan;
    if (!step_plan.ok || step_plan.dispatch_count == 0u ||
        !add(reservation.dispatch_count, step_plan.dispatch_count)) {
      return rund::AccelCheck{false, step_plan.reason == nullptr
                                         ? "compute_pipeline_capacity"
                                         : step_plan.reason};
    }
    const std::uint64_t passes = primitive_pass_count(step);
    const rund::AccelCheck structured =
        shape.plan_step(step, step_plan, nullptr, route.views,
                        shape.max_dispatch_groups, reservation);
    if (passes == 0u || !structured.ok) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    if (step.kind() != rund::kernel::NodeKind::Map) {
      if (!add(reservation.route_native_bytes, step_plan.staging_bytes)) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      continue;
    }

    const rund::kernel::ComputePlan map = step_plan;
    if (!map.ok || map.dispatch_count == 0u ||
        map.dispatch_count > std::numeric_limits<std::uint32_t>::max()) {
      return rund::AccelCheck{
          false, map.reason == nullptr ? "compute_plan_invalid" : map.reason};
    }
    std::uint64_t window_bytes = 0u;
    if (!product(map.dispatch_count,
                 sizeof(rund::kernel::ComputeDispatchWindow), window_bytes) ||
        !add(reservation.route_host_bytes, window_bytes) ||
        !add(reservation.route_native_bytes, map.param_bytes) ||
        !add(reservation.route_native_allocation_count, 1u)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }

    const bool controlled = step.control.has_count() ||
                            step.control.has_predicate() ||
                            !step.artifact.metadata.read_routes.empty();
    if (controlled) {
      std::uint64_t indirect_bytes = 0u;
      if (!product(map.dispatch_count, 4u * sizeof(std::uint32_t),
                   indirect_bytes) ||
          !add(reservation.route_native_bytes, indirect_bytes) ||
          !add(reservation.route_native_bytes, 2u * sizeof(std::uint32_t)) ||
          !add(reservation.route_native_allocation_count, 2u)) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
    }
  }
  return rund::AccelCheck{true, "ok"};
}

} // namespace rund::node::accel::detail::backend_template_plan
