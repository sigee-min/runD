#include "../local.hpp"

#include <algorithm>
#include <limits>
#include <math32/math32.hpp>
#include <span>

namespace rund::kernel::dispatch::detail {

void CopyWorkerStats(const WorkerStats& stats,
                     Telemetry& telemetry) {
  telemetry.worker_count = stats.worker_count;
  telemetry.participating_workers = stats.participating_workers;
  telemetry.total_partitions_executed = stats.total_partitions_executed;
  telemetry.worker_tile_count = stats.total_partitions_executed;
  telemetry.worker_idle_slots = stats.worker_count > stats.participating_workers
                                    ? stats.worker_count - stats.participating_workers
                                    : 0u;
  telemetry.claim_fetch_count = stats.claim_fetch_count;
  telemetry.claim_fetch_count_measured = stats.claim_fetch_count_measured;
  telemetry.claim_cost_ns = stats.claim_cost_ns;
  telemetry.claim_cost_measured = stats.claim_cost_measured;
  telemetry.static_tile_map_used = stats.static_tile_map_used;
  telemetry.global_claim_sync_elided = stats.global_claim_sync_elided;
  telemetry.dispatch_submit_cost_ns = stats.dispatch_submit_cost_ns;
  telemetry.dispatch_submit_cost_measured = stats.dispatch_submit_cost_measured;
  telemetry.dispatch_wake_to_first_worker_ns = stats.dispatch_wake_to_first_worker_ns;
  telemetry.dispatch_wake_to_last_worker_ns = stats.dispatch_wake_to_last_worker_ns;
  telemetry.dispatch_worker_wake_measured = stats.dispatch_worker_wake_measured;
  telemetry.dispatch_join_wait_ns = stats.dispatch_join_wait_ns;
  telemetry.dispatch_join_wait_measured = stats.dispatch_join_wait_measured;
  telemetry.worker_start_skew_ns = stats.worker_start_skew_ns;
  telemetry.worker_finish_skew_ns = stats.worker_finish_skew_ns;
  telemetry.barrier_wait_ns = stats.barrier_wait_ns;
  telemetry.worker_elapsed_min_ns = stats.worker_elapsed_min_ns;
  telemetry.worker_elapsed_max_ns = stats.worker_elapsed_max_ns;
  telemetry.worker_elapsed_imbalance_milli = stats.worker_elapsed_imbalance_milli;
  telemetry.worker_timing_measured = stats.worker_timing_measured;
  telemetry.slowest_worker_index = stats.slowest_worker_index;
  telemetry.slowest_worker_partitions = stats.slowest_worker_partitions;
  telemetry.slowest_worker_elapsed_ns = stats.slowest_worker_elapsed_ns;
  telemetry.root_worker_elapsed_ns = stats.root_worker_elapsed_ns;
  telemetry.root_worker_tail_wait_ns = stats.root_worker_tail_wait_ns;
  telemetry.worker_tail_attribution_measured = stats.worker_tail_attribution_measured;
  std::span<const u32> partitions_per_worker{};
  if (!stats.partitions_per_worker_sink.empty()) {
    partitions_per_worker = stats.partitions_per_worker_sink;
    telemetry.partitions_per_worker.clear();
  } else {
    partitions_per_worker =
        std::span<const u32>(stats.partitions_per_worker.data(), stats.partitions_per_worker.size());
    telemetry.partitions_per_worker = stats.partitions_per_worker;
    telemetry.worker_start_offset_ns = stats.worker_start_offset_ns;
    telemetry.worker_elapsed_ns = stats.worker_elapsed_ns;
    telemetry.worker_tail_wait_ns = stats.worker_tail_wait_ns;
  }
  if (partitions_per_worker.empty()) {
    return;
  }
  u32 min_count = std::numeric_limits<u32>::max();
  u32 max_count = 0u;
  u64 total_count = 0u;
  for (const u32 count : partitions_per_worker) {
    min_count = std::min<u32>(min_count, count);
    max_count = std::max<u32>(max_count, count);
    total_count += count;
  }
  telemetry.min_partitions_per_worker =
      min_count == std::numeric_limits<u32>::max() ? 0u : min_count;
  telemetry.max_partitions_per_worker = max_count;
  telemetry.min_tiles_per_worker = telemetry.min_partitions_per_worker;
  telemetry.max_tiles_per_worker = telemetry.max_partitions_per_worker;
  const u32 mean_count = static_cast<u32>(total_count / partitions_per_worker.size());
  telemetry.worker_partition_imbalance_milli =
      mean_count == 0u || max_count <= mean_count
          ? 0u
          : math32::detail::ScalarSatMilliRatio(max_count - mean_count, mean_count);
  telemetry.tile_imbalance_milli = telemetry.worker_partition_imbalance_milli;
}

} // namespace rund::kernel::dispatch::detail
