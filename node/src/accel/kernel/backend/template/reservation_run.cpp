#include "reservation_internal.hpp"

#include "../../prepared/template/registry.hpp"
#include "../run.hpp"

#include <kernel/core/checked.hpp>

#include <cstddef>

namespace rund::node::accel::detail::backend_template_plan {

[[nodiscard]] rund::AccelCheck
plan(const BackendRun &run, const BackendShape shape,
     PreparedKernelRouteReservation &reservation) noexcept {
  reservation = {};
  if (run.pick == nullptr || run.steps == nullptr || run.step_count == 0u ||
      shape.storage_alignment == 0u || shape.max_dispatch_groups == 0u ||
      shape.reset_dispatch_window == 0u || shape.template_capacity == 0u ||
      shape.route_header_bytes == 0u || shape.route_step_bytes == 0u ||
      shape.route_inline_step_capacity == 0u ||
      shape.template_header_bytes == 0u || shape.template_step_bytes == 0u ||
      shape.template_step_capacity == 0u || shape.plan_step == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  std::uint64_t route_steps = 0u;
  std::uint64_t template_steps = 0u;
  if ((run.step_count > shape.route_inline_step_capacity &&
       !product(run.step_count, shape.route_step_bytes, route_steps)) ||
      !product(run.step_count, shape.template_step_bytes, template_steps)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  reservation.route_host_bytes = shape.route_header_bytes;
  reservation.template_host_bytes = shape.template_header_bytes;
  if (!add(reservation.route_host_bytes, route_steps) ||
      !add(reservation.template_host_bytes, template_steps)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  reservation.route_step_count = run.step_count;
  reservation.template_step_count = run.step_count;
  reservation.template_step_capacity = shape.template_step_capacity;
  reservation.template_capacity = shape.template_capacity;
  reservation.dispatch_count = run.final_dispatch_count;
  reservation.route_native_allocation_count = 1u;
  reservation.template_native_allocation_count = 1u;
  if (reservation.dispatch_count == 0u) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }

  if (run.resets != nullptr) {
    for (const BoundReset &reset : *run.resets) {
      if (!add(reservation.reset_dispatch_count,
               reset::Commands(reset.range().count(),
                               shape.reset_dispatch_window))) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
    }
  }

  for (std::size_t index = 0u; index < run.step_count; ++index) {
    const BoundStep &bound = run.steps[index];
    if (bound.step == nullptr || bound.planned == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    const std::uint64_t passes = primitive_pass_count(*bound.step);
    const rund::AccelCheck structured =
        shape.plan_step(*bound.step, bound.planned->plan, &bound, run.views,
                        shape.max_dispatch_groups, reservation);
    if (passes == 0u || !structured.ok) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    if (bound.step->kind() != rund::kernel::NodeKind::Map) {
      if (!add(reservation.route_native_bytes,
               bound.planned->plan.staging_bytes)) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      continue;
    }

    const rund::kernel::ComputePlan &map = bound.planned->plan;
    const std::uint64_t windows = bound.planned->windows.size();
    const rund::kernel::BindingSet bindings = MapBindingFor(bound);
    if (!map.ok || windows == 0u || !bindings.ok ||
        map.input_buffer_count != bindings.resident_inputs.count ||
        map.output_buffer_count != bindings.resident_outputs.count) {
      return rund::AccelCheck{false, "compute_plan_invalid"};
    }
    std::uint64_t window_bytes = 0u;
    if (!product(windows, sizeof(rund::kernel::ComputeDispatchWindow),
                 window_bytes) ||
        !add(reservation.route_host_bytes, window_bytes) ||
        !add(reservation.route_native_bytes, map.param_bytes) ||
        !add(reservation.route_native_allocation_count, 1u)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    const bool controlled = bound.control.active() ||
                            !bound.step->artifact.metadata.read_routes.empty();
    if (controlled) {
      std::uint64_t indirect_bytes = 0u;
      if (!product(windows, 4u * sizeof(std::uint32_t), indirect_bytes) ||
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
