#include "local.hpp"

namespace rund::kernel {
namespace {

bool HasCapacityPairs(const KernelProgramCapacityProof& capacity) {
  return capacity.required.schedule_partition_capacity != 0u ||
         capacity.available.schedule_partition_capacity != 0u ||
         capacity.required.packet_capacity != 0u ||
         capacity.available.packet_capacity != 0u ||
         capacity.required.fold_slot_capacity != 0u ||
         capacity.available.fold_slot_capacity != 0u ||
         capacity.required.worker_stats_capacity != 0u ||
         capacity.available.worker_stats_capacity != 0u;
}

} // namespace

KernelProgramTelemetrySchema BuildKernelProgramTelemetrySchema(
    const Telemetry& telemetry,
    const KernelProgramCapacityProof& capacity,
    const KernelProgramPlacementMetadata& placement,
    const ScheduleView schedule) {
  const bool has_capacity_pairs = HasCapacityPairs(capacity);
  return KernelProgramTelemetrySchema{
      .useful_width = telemetry.useful_width != 0u || schedule.useful_width != 0u,
      .partition_imbalance = telemetry.partition_count != 0u || schedule.partition_count != 0u,
      .worker_participation =
          telemetry.worker_count != 0u || telemetry.total_partitions_executed != 0u,
      .static_tile_map = telemetry.static_tile_map_used,
      .global_claim_sync_elided = telemetry.global_claim_sync_elided,
      .claim_fetch_count = telemetry.claim_fetch_count_measured,
      .worker_timing = telemetry.worker_timing_measured,
      .claim_cost = telemetry.claim_cost_measured,
      .fold_cost = telemetry.fold_cost_measured,
      .capacity_proof = capacity.checked || has_capacity_pairs,
      .compile_cost_measured = telemetry.compile_cost_measured,
      .schedule_compile_cost_measured = telemetry.schedule_compile_cost_measured,
      .program_compile_cost_measured = telemetry.program_compile_cost_measured,
      .capacity_check_cost_measured = telemetry.capacity_check_cost_measured,
      .placement_cost_measured = telemetry.placement_cost_measured,
      .fold_graph_compile_cost_measured = telemetry.fold_graph_compile_cost_measured,
      .dispatch_cost_measured = telemetry.dispatch_cost_measured,
      .backend_execution_cost_measured = telemetry.backend_execution_cost_measured,
      .telemetry_update_cost_measured = telemetry.telemetry_update_cost_measured,
      .dispatch_submit_cost_measured = telemetry.dispatch_submit_cost_measured,
      .dispatch_worker_wake_measured = telemetry.dispatch_worker_wake_measured,
      .dispatch_join_wait_measured = telemetry.dispatch_join_wait_measured,
      .worker_tail_attribution_measured = telemetry.worker_tail_attribution_measured,
      .claim_cost_measured = telemetry.claim_cost_measured,
      .fold_cost_measured = telemetry.fold_cost_measured,
      .capacity_checked = capacity.checked,
      .capacity_satisfied = capacity.satisfied,
      .capacity_required_available = capacity.checked && has_capacity_pairs,
      .no_alloc_capacity_margin = capacity.checked,
      .placement_work_units =
          placement.has_packet_work_units ||
          placement.max_partition_work_units != 0u ||
          placement.min_partition_work_units != 0u ||
          placement.work_imbalance_milli != 0u,
      .placement_worker_capacity = placement.has_worker_capacity,
      .placement_locality =
          placement.has_packet_hints ||
          placement.locality_bucket_crossing_count != 0u,
      .strict_fp_software_reference = telemetry.strict_fp_software_reference,
      .strict_fp_backend_supported = telemetry.strict_fp_backend_supported,
      .work_imbalance_measured = telemetry.work_imbalance_measured,
      .locality_bucket_crossing_measured = telemetry.locality_bucket_crossing_measured,
      .no_alloc_capacity_margin_measured = telemetry.no_alloc_capacity_margin_measured,
  };
}

} // namespace rund::kernel
