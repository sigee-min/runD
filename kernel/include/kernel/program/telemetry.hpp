#pragma once

namespace rund::kernel {

struct KernelProgramTelemetrySchema {
  bool useful_width = false;
  bool partition_imbalance = false;
  bool worker_participation = false;
  bool static_tile_map = false;
  bool global_claim_sync_elided = false;
  bool claim_fetch_count = false;
  bool worker_timing = false;
  bool claim_cost = false;
  bool fold_cost = false;
  bool capacity_proof = false;
  bool compile_cost_measured = false;
  bool schedule_compile_cost_measured = false;
  bool program_compile_cost_measured = false;
  bool capacity_check_cost_measured = false;
  bool placement_cost_measured = false;
  bool fold_graph_compile_cost_measured = false;
  bool dispatch_cost_measured = false;
  bool backend_execution_cost_measured = false;
  bool telemetry_update_cost_measured = false;
  bool dispatch_submit_cost_measured = false;
  bool dispatch_worker_wake_measured = false;
  bool dispatch_join_wait_measured = false;
  bool worker_tail_attribution_measured = false;
  bool claim_cost_measured = false;
  bool fold_cost_measured = false;
  bool capacity_checked = false;
  bool capacity_satisfied = false;
  bool capacity_required_available = false;
  bool no_alloc_capacity_margin = false;
  bool placement_work_units = false;
  bool placement_worker_capacity = false;
  bool placement_locality = false;
  bool strict_fp_software_reference = false;
  bool strict_fp_backend_supported = false;
  bool work_imbalance_measured = false;
  bool locality_bucket_crossing_measured = false;
  bool no_alloc_capacity_margin_measured = false;
  bool memory_pass_count = false;
  bool memory_unit_bytes = false;
  bool store_behavior = false;
  bool working_set_bytes = false;
  bool tile_cache_fit = false;
  bool tlb_page_footprint = false;
};

} // namespace rund::kernel
