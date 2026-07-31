#pragma once

#include <kernel/dispatch/worker/enums.hpp>
#include <kernel/schedule/planner/policy.hpp>

#include <vector>

namespace rund::kernel {

struct Telemetry {
  u32 packet_count = 0;
  u32 execution_width = 0;
  u32 useful_width = 0;
  u32 partition_count = 0;
  u32 worker_slot_count = 0;
  u32 fold_slot_count = 0;
  u32 alignment_packets = 1;
  u32 packets_per_partition_max = 0;
  u32 max_partition_size = 0;
  u32 min_partition_size = 0;
  u32 partition_size_imbalance_milli = 0;
  u32 worker_count = 0;
  u32 participating_workers = 0;
  u32 total_partitions_executed = 0;
  u32 max_partitions_per_worker = 0;
  u32 min_partitions_per_worker = 0;
  u32 worker_partition_imbalance_milli = 0;
  u32 worker_idle_slots = 0;
  u32 worker_tile_count = 0;
  u32 min_tiles_per_worker = 0;
  u32 max_tiles_per_worker = 0;
  u32 tile_imbalance_milli = 0;
  u64 max_partition_work_units = 0;
  u64 min_partition_work_units = 0;
  u32 work_imbalance_milli = 0;
  u32 locality_bucket_crossing_count = 0;
  bool has_worker_capacity = false;
  bool worker_capacity_truth = false;
  u32 worker_capacity_imbalance_milli = 0;
  WorkerTruthLevel affinity_truth_level = WorkerTruthLevel::Unknown;
  bool affinity_hint_only = false;
  bool strict_fp_software_reference = false;
  // Capability evidence only; not proof that a hardware strict-FP path ran.
  bool strict_fp_backend_supported = false;
  u32 claim_fetch_count = 0;
  bool claim_fetch_count_measured = false;
  bool static_tile_map_used = false;
  bool global_claim_sync_elided = false;
  u32 backend_dispatch_count = 0;
  u64 compile_cost_ns = 0;
  bool compile_cost_measured = false;
  u64 schedule_compile_cost_ns = 0;
  bool schedule_compile_cost_measured = false;
  u64 program_compile_cost_ns = 0;
  bool program_compile_cost_measured = false;
  u64 capacity_check_cost_ns = 0;
  bool capacity_check_cost_measured = false;
  u64 placement_cost_ns = 0;
  bool placement_cost_measured = false;
  u64 fold_graph_compile_cost_ns = 0;
  bool fold_graph_compile_cost_measured = false;
  u64 dispatch_cost_ns = 0;
  bool dispatch_cost_measured = false;
  u64 backend_execution_cost_ns = 0;
  bool backend_execution_cost_measured = false;
  u64 telemetry_update_cost_ns = 0;
  bool telemetry_update_cost_measured = false;
  u64 claim_cost_ns = 0;
  bool claim_cost_measured = false;
  u64 dispatch_submit_cost_ns = 0;
  bool dispatch_submit_cost_measured = false;
  u64 dispatch_wake_to_first_worker_ns = 0;
  u64 dispatch_wake_to_last_worker_ns = 0;
  bool dispatch_worker_wake_measured = false;
  u64 dispatch_join_wait_ns = 0;
  bool dispatch_join_wait_measured = false;
  u64 worker_start_skew_ns = 0;
  u64 worker_finish_skew_ns = 0;
  u64 barrier_wait_ns = 0;
  u64 worker_elapsed_min_ns = 0;
  u64 worker_elapsed_max_ns = 0;
  u32 worker_elapsed_imbalance_milli = 0;
  bool worker_timing_measured = false;
  u32 slowest_worker_index = 0;
  u32 slowest_worker_partitions = 0;
  u64 slowest_worker_elapsed_ns = 0;
  u64 root_worker_elapsed_ns = 0;
  u64 root_worker_tail_wait_ns = 0;
  bool worker_tail_attribution_measured = false;
  u64 fold_cost_ns = 0;
  bool fold_cost_measured = false;
  bool capacity_checked = false;
  bool capacity_satisfied = false;
  u32 required_schedule_partition_capacity = 0;
  u32 available_schedule_partition_capacity = 0;
  u32 required_packet_capacity = 0;
  u32 available_packet_capacity = 0;
  u32 required_fold_slot_capacity = 0;
  u32 available_fold_slot_capacity = 0;
  u32 required_worker_stats_capacity = 0;
  u32 available_worker_stats_capacity = 0;
  u32 no_alloc_capacity_margin = 0;
  bool work_imbalance_measured = false;
  bool locality_bucket_crossing_measured = false;
  bool no_alloc_capacity_margin_measured = false;
  PlacementPolicy placement = PlacementPolicy::Uniform;
  bool no_allocation = false;
  std::vector<u32> partitions_per_worker{};
  std::vector<u64> worker_start_offset_ns{};
  std::vector<u64> worker_elapsed_ns{};
  std::vector<u64> worker_tail_wait_ns{};
};

struct Result {
  bool ok = false;
  const char* failure_reason = "not_run";
  Telemetry telemetry{};
};

} // namespace rund::kernel
