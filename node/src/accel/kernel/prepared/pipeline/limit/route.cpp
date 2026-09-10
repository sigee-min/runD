#include "internal.hpp"

#include "../../../../context/internal/support.hpp"
#include "../../../backend/template/identity.hpp"
#include "../../../recurrence.hpp"
#include "../../../recurrence/plan.hpp"
#include "../../evidence.hpp"
#include "../recurrence.hpp"
#include "../reservation.hpp"
#include "../structure.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>

namespace rund::node::accel::detail::prepared_pipeline_limit {
namespace {

using ::rund::kernel::checked::mul;

} // namespace

bool plan_route(const rund::AccelContext &context,
                const std::span<const PreparedKernelProgramRoute> routes,
                const std::size_t route_index,
                const PreparedKernelPipelineShape shape,
                State &state) noexcept {
  PreparedKernelPipelineReservation &result = state.result;
  const PreparedKernelProgramRoute &route = routes[route_index];
  if (route.kernel == nullptr || route.tile_count == 0u ||
      route.entry_count == 0u ||
      route.window_count > route.occurrence_count ||
      route.nested_group_count > route.entry_count ||
      route.map_recurrence_history_group_count >
          route.map_recurrence_group_count ||
      route.route_copies == 0u || route.route_copies > 2u ||
      route.route_copies != shape.route_copies) {
    result.reason = "accel_kernel_run_invalid";
    return false;
  }
  const KernelExecution execution = AdmitKernelForExecution(context, *route.kernel);
  const BackendOps *const candidate =
      execution.context_admission.pick == nullptr
          ? nullptr
          : execution.context_admission.pick->ops;
  if (!execution.admission.check.ok || candidate == nullptr ||
      candidate->api != context.api ||
      candidate->plan_pipeline_program == nullptr ||
      candidate->same_pipeline_program_template == nullptr ||
      (state.ops != nullptr && state.ops != candidate)) {
    result.reason = execution.admission.check.reason == nullptr
                        ? "accel_kernel_run_invalid"
                        : execution.admission.check.reason;
    return false;
  }
  state.ops = candidate;
  state.stream_count =
      std::max<std::uint64_t>(state.stream_count, route.route_copies);
  std::uint64_t entries = 0u;
  if (!mul(route.entry_count, route.route_copies, entries) ||
      !accumulate(state.entry_count, entries)) {
    return false;
  }
  fingerprint_route(result.fingerprint_hi, result.fingerprint_lo,
                    route.kernel->kernel_id, route.kernel->graph_id_hi,
                    route.kernel->graph_id_lo, route.kernel->node_count,
                    route.kernel->api, route.tile_count, route.views,
                    route.scratch, route.entry_count, route.occurrence_count,
                    route.window_count, route.nested_group_count,
                    route.map_recurrence_group_count,
                    route.map_recurrence_history_group_count,
                    route.recurrence_fingerprint_hi,
                    route.recurrence_fingerprint_lo, route.route_copies);
  const backend_template_plan::MapSpecializationFingerprint map_fingerprint =
      backend_template_plan::program_map_specialization_fingerprint(execution,
                                                                    route);
  if (!map_fingerprint.ok) {
    result.reason = "accel_kernel_run_invalid";
    return false;
  }
  fingerprint_mix(result.fingerprint_hi, map_fingerprint.hi);
  fingerprint_mix(result.fingerprint_lo, map_fingerprint.lo);

  PreparedKernelRouteReservation planned{};
  const rund::AccelCheck checked =
      candidate->plan_pipeline_program(execution, route, planned);
  if (!checked.ok || planned.route_step_count == 0u ||
      planned.template_step_count == 0u || planned.template_capacity == 0u ||
      !accumulate(planned.route_host_bytes, sizeof(prepared::RunState))) {
    result.reason = checked.reason == nullptr ? "compute_pipeline_capacity"
                                              : checked.reason;
    return false;
  }
  if (!accumulate_route_projection(state.backend_projection, planned,
                                   route.entry_count, route.occurrence_count,
                                   route.window_count)) {
    result.reason = "compute_pipeline_capacity";
    return false;
  }
  result.template_capacity =
      std::min(result.template_capacity, planned.template_capacity);
  result.template_step_capacity =
      std::min(result.template_step_capacity, planned.template_step_capacity);
  std::uint64_t route_host = 0u;
  std::uint64_t route_native = 0u;
  std::uint64_t route_steps = 0u;
  std::uint64_t descriptor_sets = 0u;
  std::uint64_t descriptors = 0u;
  std::uint64_t route_allocations = 0u;
  if (!mul(planned.route_host_bytes, route.route_copies, route_host) ||
      !mul(planned.route_native_bytes, route.route_copies, route_native) ||
      !mul(planned.route_step_count, route.route_copies, route_steps) ||
      !mul(planned.descriptor_set_count, route.route_copies, descriptor_sets) ||
      !mul(planned.descriptor_count, route.route_copies, descriptors) ||
      !mul(planned.route_native_allocation_count, route.route_copies,
           route_allocations) ||
      !accumulate(result.route_count, route.route_copies) ||
      !mul(route.entry_count, route.route_copies, entries) ||
      !accumulate(result.authored_entry_count, entries) ||
      !mul(route.occurrence_count, route.route_copies, entries) ||
      !accumulate(result.occurrence_count, entries) ||
      !mul(route.window_count, route.route_copies, entries) ||
      !accumulate(result.window_count, entries) ||
      !mul(route.nested_group_count, route.route_copies, entries) ||
      !accumulate(result.nested_group_count, entries) ||
      !accumulate(result.route_host_bytes, route_host) ||
      !accumulate(result.route_native_bytes, route_native) ||
      !accumulate(result.route_step_count, route_steps) ||
      !accumulate(result.descriptor_set_count, descriptor_sets) ||
      !accumulate(result.descriptor_count, descriptors) ||
      !accumulate(result.native_allocation_count, route_allocations)) {
    return false;
  }

  MapRecurrencePreparationPlan recurrence_plan =
      PlanMapRecurrencePreparation(execution, route);
  MapRecurrenceTemplateCapacities recurrence_template_capacities{};
  if (!recurrence_plan.ok ||
      !program_recurrence_template_capacities(
          execution, routes, route_index, recurrence_plan,
          recurrence_template_capacities)) {
    result.reason = recurrence_plan.reason == nullptr
                        ? "compute_pipeline_capacity"
                        : recurrence_plan.reason;
    return false;
  }
  recurrence_plan.terminal_template_group_capacity =
      recurrence_plan.terminal_group_count() == 0u
          ? 0u
          : recurrence_template_capacities.terminal;
  recurrence_plan.history_template_group_capacity =
      recurrence_plan.history_group_count == 0u
          ? 0u
          : recurrence_template_capacities.history;
  PreparedMapRecurrenceReservation recurrence{};
  const rund::AccelCheck recurrence_checked =
      plan_map_recurrence_reservation(*candidate, recurrence_plan, recurrence);
  PreparedMapRecurrenceReservation terminal_recurrence{};
  PreparedMapRecurrenceReservation history_recurrence{};
  const rund::AccelCheck recurrence_variants_checked =
      recurrence_checked.ok
          ? verify_map_recurrence_template_variants(
                *candidate, recurrence_plan, recurrence, terminal_recurrence,
                history_recurrence)
          : recurrence_checked;
  PreparedMapRecurrenceReservation scaled_recurrence_route{};
  PreparedKernelPipelineReservation recurrence_route_projection{};
  if (!recurrence_checked.ok || !recurrence_variants_checked.ok ||
      !scale_map_recurrence_route_reservation(
          recurrence, route.route_copies, scaled_recurrence_route) ||
      !project_map_recurrence_reservation(scaled_recurrence_route,
                                          recurrence_route_projection) ||
      !accumulate_reservation(result, recurrence_route_projection)) {
    result.reason = !recurrence_checked.ok
                        ? (recurrence_checked.reason == nullptr
                               ? "compute_pipeline_capacity"
                               : recurrence_checked.reason)
                        : (recurrence_variants_checked.reason == nullptr
                               ? "compute_pipeline_capacity"
                               : recurrence_variants_checked.reason);
    return false;
  }
  const auto recurrence_variant_seen = [&](const bool history) noexcept {
    for (std::size_t prior = 0u; prior < route_index; ++prior) {
      const PreparedKernelProgramRoute &previous = routes[prior];
      if (!same_program_recurrence_authority(route, previous)) {
        continue;
      }
      const MapRecurrencePreparationPlan previous_plan =
          PlanMapRecurrencePreparation(execution, previous);
      if (SameMapRecurrenceTemplate(recurrence_plan, previous_plan, history)) {
        return true;
      }
    }
    return false;
  };
  const auto accumulate_recurrence_template =
      [&](const bool present, const bool history,
          const PreparedMapRecurrenceReservation &reservation) {
        if (!present || recurrence_variant_seen(history)) {
          return true;
        }
        PreparedKernelPipelineReservation recurrence_template_projection{};
        return project_map_recurrence_reservation(
                   reservation, recurrence_template_projection) &&
               accumulate_reservation(result,
                                      recurrence_template_projection);
      };
  if (!accumulate_recurrence_template(
          recurrence_plan.terminal_group_count() != 0u, false,
          terminal_recurrence) ||
      !accumulate_recurrence_template(
          recurrence_plan.history_group_count != 0u, true,
          history_recurrence)) {
    result.reason = "compute_pipeline_capacity";
    return false;
  }
  for (std::size_t prior = 0u; prior < route_index; ++prior) {
    if (candidate->same_pipeline_program_template(execution, route,
                                                  routes[prior])) {
      return true;
    }
  }
  if (!accumulate(result.template_count, 1u) ||
      !accumulate(result.template_host_bytes, planned.template_host_bytes) ||
      !accumulate(result.template_native_bytes, planned.template_native_bytes) ||
      !accumulate(result.template_source_bytes, planned.template_source_bytes) ||
      !accumulate(result.template_step_count, planned.template_step_count) ||
      !accumulate(result.native_allocation_count,
                  planned.template_native_allocation_count)) {
    return false;
  }
  result.source_transient_bytes =
      std::max(result.source_transient_bytes, planned.source_transient_bytes);
  return true;
}

} // namespace rund::node::accel::detail::prepared_pipeline_limit
