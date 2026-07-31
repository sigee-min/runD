#pragma once

#include "../pool.hpp"

#include <span>

namespace kernel_contract_test {

inline void PublishFakeStats(const rund::kernel::u32 worker_count,
                             const std::span<const rund::kernel::u32> partitions_per_worker,
                             rund::kernel::WorkerStats* const out_stats) {
  if (out_stats == nullptr) {
    return;
  }
  out_stats->worker_count = worker_count;
  out_stats->static_tile_map_used = true;
  out_stats->global_claim_sync_elided = true;
  out_stats->claim_fetch_count = 0u;
  out_stats->claim_fetch_count_measured = true;
  out_stats->claim_cost_measured = false;
  if (!out_stats->partitions_per_worker_sink.empty() &&
      out_stats->partitions_per_worker_sink.size() >= partitions_per_worker.size()) {
    for (std::size_t index = 0u; index < partitions_per_worker.size(); ++index) {
      out_stats->partitions_per_worker_sink[index] = partitions_per_worker[index];
    }
    out_stats->no_allocation_sink_used = true;
    out_stats->partitions_per_worker.clear();
  } else {
    out_stats->partitions_per_worker.assign(partitions_per_worker.begin(), partitions_per_worker.end());
  }
  out_stats->participating_workers = 0u;
  out_stats->total_partitions_executed = 0u;
  for (const rund::kernel::u32 partition_count : partitions_per_worker) {
    out_stats->total_partitions_executed += partition_count;
    if (partition_count != 0u) {
      out_stats->participating_workers += 1u;
    }
  }
}

} // namespace kernel_contract_test
