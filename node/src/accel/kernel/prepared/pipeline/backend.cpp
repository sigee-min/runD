#include "../run.hpp"
#include "backend.hpp"

#include "structure.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelCheck finalize_pipeline_backend_structure(
    const rund::AccelContext &context, const BackendOps &ops,
    const PreparedKernelPipelineReservation &projection,
    const std::uint64_t publication_count,
    const std::uint64_t terminal_publication_count,
    const std::uint64_t publication_command_count,
    const std::uint64_t window_state_count,
    const std::uint64_t window_descriptor_state_count,
    const std::uint64_t profile_step_count,
    const std::uint64_t profile_command_count,
    PreparedKernelPipelineReservation &result) noexcept {
  if (ops.plan_pipeline_structure == nullptr ||
      projection.occurrence_count == 0u ||
      projection.backend_dispatch_count == 0u ||
      projection.backend_step_occurrence_count == 0u ||
      projection.backend_step_description_count == 0u) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  result.backend_dispatch_count = projection.backend_dispatch_count;
  result.host_transient_bytes = projection.host_transient_bytes;
  result.backend_reset_dispatch_count = projection.backend_reset_dispatch_count;
  result.backend_step_occurrence_count =
      projection.backend_step_occurrence_count;
  result.backend_step_description_count =
      projection.backend_step_description_count;
  result.backend_status_entry_count = projection.backend_status_entry_count;
  result.backend_window_dispatch_count =
      projection.backend_window_dispatch_count;
  result.backend_indirect_dispatch_count =
      projection.backend_indirect_dispatch_count;
  result.backend_window_state_count = window_state_count;
  result.backend_window_descriptor_state_count = window_descriptor_state_count;
  result.backend_status_source_count = projection.backend_status_source_count;
  result.backend_telemetry_count = projection.backend_telemetry_count;
  result.backend_status_command_count = projection.backend_status_command_count;
  result.backend_telemetry_command_count =
      projection.backend_telemetry_command_count;
  result.backend_parameter_bytes = projection.backend_parameter_bytes;
  result.backend_publication_count = publication_count;
  result.backend_terminal_publication_count = terminal_publication_count;
  result.backend_publication_command_count = publication_command_count;
  result.backend_command_binding_slot_upper =
      projection.backend_command_binding_slot_upper;
  result.backend_profile_step_count = profile_step_count;
  result.backend_profile_command_count = profile_command_count;
  return ops.plan_pipeline_structure(context, result);
}

[[nodiscard]] rund::AccelCheck plan_runtime_backend_structure(
    const rund::AccelContext &context,
    const std::span<const PreparedKernelRun *const> runs,
    const std::span<const BackendRecurrence> recurrences,
    const std::span<const BackendPublish> publications,
    const std::uint64_t profile_step_count,
    const std::uint64_t profile_command_count,
    PreparedKernelPipelineReservation &structure) noexcept {
  if (runs.empty() || runs.size() != recurrences.size()) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  const BackendOps *ops = nullptr;
  PreparedKernelPipelineReservation projection{};
  for (std::size_t index = 0u; index < runs.size(); ++index) {
    const PreparedKernelRun *const item = runs[index];
    const auto *const state =
        item == nullptr
            ? nullptr
            : static_cast<const prepared::RunState *>(item->owner.get());
    const BackendOps *const candidate =
        state == nullptr ? nullptr : state->bound.run.ops;
    PreparedKernelRouteReservation route{};
    if (item == nullptr || !item->ok || state == nullptr ||
        candidate == nullptr || candidate->plan_pipeline_private == nullptr ||
        (ops != nullptr && ops != candidate)) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    bool duplicate_route = false;
    for (std::size_t prior = 0u; prior < index; ++prior) {
      const PreparedKernelRun *const previous = runs[prior];
      if (previous != nullptr && previous->owner.get() == item->owner.get()) {
        duplicate_route = true;
        break;
      }
    }
    ops = candidate;
    if (duplicate_route) {
      continue;
    }
    const rund::AccelCheck planned =
        candidate->plan_pipeline_private(state->bound.run, route);
    std::uint64_t entry_count = 0u;
    std::uint64_t occurrence_count = 0u;
    std::uint64_t window_count = 0u;
    std::uint64_t nested_group_count = 0u;
    std::uint64_t map_recurrence_group_count = 0u;
    std::uint64_t map_recurrence_history_group_count = 0u;
    std::uint64_t recurrence_hi = 0u;
    std::uint64_t recurrence_lo = 0u;
    if (!planned.ok) {
      return rund::AccelCheck{false, planned.reason == nullptr
                                         ? "compute_pipeline_capacity"
                                         : planned.reason};
    }
    if (!route_recurrence_shape(
            runs, recurrences, item->owner.get(), entry_count, occurrence_count,
            window_count, nested_group_count, map_recurrence_group_count,
            map_recurrence_history_group_count, recurrence_hi, recurrence_lo)) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    if (!accumulate_route_projection(projection, route, entry_count,
                                     occurrence_count, window_count)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }
  const PreparedKernelPipelineShape shape =
      runtime_pipeline_shape(publications, recurrences, 1u, 1u, false);
  if (ops == nullptr ||
      projection.occurrence_count != structure.occurrence_count) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  return finalize_pipeline_backend_structure(
      context, *ops, projection, publications.size(),
      shape.terminal_publication_count, shape.backend_publication_command_count,
      shape.window_state_count, shape.window_descriptor_state_count,
      profile_step_count, profile_command_count, structure);
}

} // namespace rund::node::accel::detail
