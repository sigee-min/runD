#include "../../../../../node/src/accel/kernel/prepared/interface/api.hpp"
#include "contract.hpp"

#include <node/runtime/compute/access.hpp>

namespace rund::measure::compute::preparation_memory {

[[nodiscard]] bool PlanContract(const ::rund::compute::PipelinePlan &plan,
                                const Backend backend) noexcept {
  const std::uint64_t prepared_components =
      plan.prepared_buffer_bytes + plan.prepared_host_bytes +
      plan.prepared_tile_bytes + plan.prepared_native_bytes;
  const bool backend_components = backend == Backend::Cpu
                                      ? plan.prepared_host_bytes != 0u &&
                                            plan.prepared_tile_bytes != 0u &&
                                            plan.prepared_native_bytes == 0u
                                      : plan.prepared_native_bytes != 0u;
  return plan.outer_window_count == Outer && plan.tile_capacity == Tile &&
         plan.inner_iteration_count == Inner &&
         plan.prepared_template_count == PreparedTemplates &&
         plan.prepared_command_count == PreparedCommands &&
         plan.node_count >= 2u * SeedScanMapPairs &&
         plan.resource_count != 0u && plan.barrier_count != 0u &&
         plan.allocation_count != 0u && plan.reuse_count != 0u &&
         plan.prepared_bytes == prepared_components &&
         plan.prepared_bytes != 0u &&
         plan.peak_bytes ==
             plan.state_bytes + plan.transient_bytes + plan.prepared_bytes &&
         plan.arena_extent_bytes <= plan.peak_bytes &&
         plan.committed_peak_bytes >= plan.peak_bytes &&
         (backend == Backend::Cpu
              ? plan.arena_extent_bytes != 0u
              : plan.arena_extent_bytes == 0u &&
                    plan.committed_peak_bytes == plan.peak_bytes) &&
         plan.total_bytes == plan.persistent_bytes + plan.peak_bytes &&
         plan.physical_bytes == plan.peak_bytes &&
         plan.logical_bytes > plan.live_bytes &&
         plan.live_bytes <= plan.physical_bytes && backend_components;
}

[[nodiscard]] bool
PreciseFailure(const PreparationMemoryObservation &observed) noexcept {
  return observed.location.native_reason_key != nullptr &&
         observed.location.step != ::rund::compute::Location::none &&
         observed.location.template_index != ::rund::compute::Location::none &&
         ::rund::compute::valid_pipeline_nested_coordinate(
             observed.location.nested_phase,
             observed.location.outer_iteration !=
                 ::rund::compute::Location::none,
             observed.location.inner_iteration !=
                 ::rund::compute::Location::none);
}

[[nodiscard]] bool
PlanObservationContract(const PreparationMemoryObservation &observed) noexcept {
  return observed.plan_contract && observed.plan_wall_us > 0.0 &&
         observed.plan_current_rss_before != 0u &&
         observed.plan_current_rss_after != 0u &&
         observed.plan_rss_before != 0u &&
         observed.plan_rss_after >= observed.plan_rss_before;
}

[[nodiscard]] bool
PreparedMemoryContract(const ::rund::compute::PipelinePlan &plan,
                       const ::rund::compute::MemoryStats &memory,
                       const Backend backend) noexcept {
  using namespace ::rund::compute;
  if (!memory.available() || memory.backend != backend ||
      memory.scope != MemoryScope::Pipeline) {
    return false;
  }

  const std::uint64_t planned_resident =
      plan.state_bytes + plan.transient_bytes + plan.prepared_buffer_bytes;
  const std::uint64_t planned_host =
      planned_resident + plan.prepared_host_bytes;
  if (memory.resident.current > plan.peak_bytes) {
    return false;
  }
  if (backend == Backend::Cpu) {
    return plan.prepared_native_bytes == 0u &&
           memory.resident.current == planned_resident &&
           memory.host.current <= planned_host &&
           memory.tile.current == plan.prepared_tile_bytes &&
           memory.host.current <= plan.peak_bytes &&
           memory.tile.current <= plan.peak_bytes - memory.host.current;
  }

  // Accelerator native reservations cover runD-owned command/descriptor
  // storage. Driver allocation granularity is deliberately telemetry-only,
  // so require a frozen nonzero reservation and observable native storage,
  // without claiming byte equality the backend cannot guarantee.
  return plan.prepared_native_bytes != 0u && memory.host.current != 0u &&
         (memory.device.current != 0u || memory.staging.current != 0u);
}


[[nodiscard]] bool CaptureLargestRetainedGroup(
    PreparationMemoryObservation &observed,
    const ::rund::compute::Pipeline &pipeline) noexcept {
  using namespace ::rund::compute;
  std::array<MemoryEntry, 16u> entries{};
  const MemorySnapshot snapshot = pipeline.memory_snapshot(entries);
  if (snapshot.summary.scope != MemoryScope::Pipeline || snapshot.truncated()) {
    return false;
  }
  const MemoryEntry *largest = nullptr;
  for (std::size_t index = 0u; index < snapshot.written; ++index) {
    const MemoryEntry &entry = entries[index];
    if (entry.bytes.current != 0u &&
        (largest == nullptr || entry.bytes.current > largest->bytes.current)) {
      largest = &entry;
    }
  }
  if (largest == nullptr) {
    return false;
  }
  observed.largest_retained_group = *largest;
  observed.retained_group_available = true;
  return true;
}

[[nodiscard]] constexpr bool
Contains(const ::rund::compute::MemoryCounter &total,
         const ::rund::node::accel::detail::PreparedMemory part) noexcept {
  return part.current <= total.current && part.peak <= total.peak &&
         part.cumulative <= total.cumulative && part.reused <= total.reused &&
         part.budget <= total.budget;
}

[[nodiscard]] bool
CaptureBackendPreparation(PreparationMemoryObservation &observed,
                          const ::rund::compute::Pipeline &pipeline,
                          const Backend backend) noexcept {
  using namespace ::rund::compute;
  if (backend == Backend::Cpu) {
    observed.backend_reservation_contract =
        observed.plan.prepared_native_bytes == 0u;
    observed.backend_telemetry_contract = true;
    return observed.backend_reservation_contract;
  }

  const std::shared_ptr<::rund::compute::detail::PipelineState> &state =
      ::rund::compute::detail::PipelineStateAccess::state(pipeline);
  if (state == nullptr || state->device == nullptr ||
      state->device->ops == nullptr ||
      state->device->ops->pipeline_memory == nullptr) {
    return false;
  }

  observed.backend_observed = true;
  observed.backend_memory = state->device->ops->pipeline_memory(*state);
  observed.backend_limit = state->accel_templates.limit;
  observed.backend_consumed = state->accel_templates.reservation;
  const auto &recurrence_limit = observed.backend_limit.map_recurrence;
  const auto &recurrence_consumed = observed.backend_consumed.map_recurrence;
  const bool recurrence_reservation_contract =
      recurrence_limit.terminal_template_group_capacity <=
          recurrence_limit.group_count &&
      recurrence_limit.history_template_group_capacity ==
          recurrence_limit.group_count -
              recurrence_limit.terminal_template_group_capacity &&
      recurrence_limit.group_count == recurrence_consumed.group_count &&
      recurrence_limit.history_group_count ==
          recurrence_consumed.history_group_count &&
      recurrence_limit.template_count == recurrence_consumed.template_count &&
      recurrence_limit.terminal_template_group_capacity ==
          recurrence_consumed.terminal_template_group_capacity &&
      recurrence_limit.history_template_group_capacity ==
          recurrence_consumed.history_template_group_capacity &&
      recurrence_limit.descriptor_set_count ==
          recurrence_consumed.descriptor_set_count &&
      recurrence_limit.descriptor_count ==
          recurrence_consumed.descriptor_count &&
      recurrence_limit.template_native_allocation_count ==
          recurrence_consumed.template_native_allocation_count;
  const bool backend_structure_contract =
      observed.backend_limit.occurrence_count ==
          observed.backend_consumed.occurrence_count &&
      observed.backend_limit.backend_dispatch_count ==
          observed.backend_consumed.backend_dispatch_count &&
      observed.backend_limit.backend_reset_dispatch_count ==
          observed.backend_consumed.backend_reset_dispatch_count &&
      observed.backend_limit.backend_window_dispatch_count ==
          observed.backend_consumed.backend_window_dispatch_count &&
      observed.backend_limit.backend_indirect_dispatch_count ==
          observed.backend_consumed.backend_indirect_dispatch_count &&
      observed.backend_limit.backend_step_occurrence_count ==
          observed.backend_consumed.backend_step_occurrence_count &&
      observed.backend_limit.backend_status_source_count ==
          observed.backend_consumed.backend_status_source_count &&
      observed.backend_limit.backend_status_entry_count ==
          observed.backend_consumed.backend_status_entry_count &&
      observed.backend_limit.backend_telemetry_count ==
          observed.backend_consumed.backend_telemetry_count &&
      observed.backend_limit.backend_status_command_count ==
          observed.backend_consumed.backend_status_command_count &&
      observed.backend_limit.backend_telemetry_command_count ==
          observed.backend_consumed.backend_telemetry_command_count &&
      observed.backend_limit.backend_window_control_command_count ==
          observed.backend_consumed.backend_window_control_command_count &&
      observed.backend_limit.backend_publication_command_count ==
          observed.backend_consumed.backend_publication_command_count &&
      observed.backend_consumed.backend_command_chunk_count <=
          observed.backend_limit.backend_command_chunk_count &&
      observed.backend_consumed.backend_command_native_bytes <=
          observed.backend_limit.backend_command_native_bytes &&
      observed.backend_limit.backend_parameter_bytes ==
          observed.backend_consumed.backend_parameter_bytes &&
      observed.backend_limit.backend_step_description_count ==
          observed.backend_consumed.backend_step_description_count;
  observed.backend_reservation_contract =
      observed.backend_limit.ok && observed.backend_consumed.ok &&
      observed.backend_limit.fingerprint_hi != 0u &&
      observed.backend_limit.fingerprint_lo != 0u &&
      observed.backend_limit.host_bytes <= observed.plan.prepared_host_bytes &&
      observed.backend_limit.native_bytes ==
          observed.plan.prepared_native_bytes &&
      recurrence_reservation_contract && backend_structure_contract &&
      ::rund::node::accel::detail::PreparedKernelPipelineReservationWithin(
          observed.backend_consumed, observed.backend_limit);

  // Pipeline-private Run resources and opaque driver allocation granularity
  // are reported by the complete public counters. The backend bridge is the
  // exact primary/alternate/registry subset, so it must reconcile into those
  // counters without pretending physical Device bytes equal logical native
  // requests.
  observed.backend_telemetry_contract =
      observed.backend_memory.host.current <=
          observed.backend_limit.host_bytes &&
      observed.backend_memory.host.peak <= observed.backend_limit.host_bytes &&
      Contains(observed.memory.host, observed.backend_memory.host) &&
      Contains(observed.memory.device, observed.backend_memory.device) &&
      Contains(observed.memory.staging, observed.backend_memory.staging) &&
      (observed.backend_memory.device.current != 0u ||
       observed.backend_memory.staging.current != 0u);
  return observed.backend_reservation_contract &&
         observed.backend_telemetry_contract;
}


} // namespace rund::measure::compute::preparation_memory
