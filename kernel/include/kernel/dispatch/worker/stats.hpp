#pragma once

#include <kernel/core/model.hpp>

#include <span>
#include <vector>

namespace rund::kernel {

struct WorkerStats {
  u32 worker_count = 0u;
  u32 participating_workers = 0u;
  u32 total_partitions_executed = 0u;
  u32 claim_fetch_count = 0u;
  bool claim_fetch_count_measured = false;
  u64 claim_cost_ns = 0u;
  bool claim_cost_measured = false;
  bool static_tile_map_used = false;
  bool global_claim_sync_elided = false;
  u64 dispatch_submit_cost_ns = 0u;
  bool dispatch_submit_cost_measured = false;
  u64 dispatch_wake_to_first_worker_ns = 0u;
  u64 dispatch_wake_to_last_worker_ns = 0u;
  bool dispatch_worker_wake_measured = false;
  u64 dispatch_join_wait_ns = 0u;
  bool dispatch_join_wait_measured = false;
  u64 worker_start_skew_ns = 0u;
  u64 worker_finish_skew_ns = 0u;
  u64 barrier_wait_ns = 0u;
  u64 worker_elapsed_min_ns = 0u;
  u64 worker_elapsed_max_ns = 0u;
  u32 worker_elapsed_imbalance_milli = 0u;
  bool worker_timing_measured = false;
  u32 slowest_worker_index = 0u;
  u32 slowest_worker_partitions = 0u;
  u64 slowest_worker_elapsed_ns = 0u;
  u64 root_worker_elapsed_ns = 0u;
  u64 root_worker_tail_wait_ns = 0u;
  bool worker_tail_attribution_measured = false;
  bool no_allocation_sink_used = false;
  std::span<u32> partitions_per_worker_sink{};
  std::span<u64> worker_start_offset_ns_sink{};
  std::span<u64> worker_elapsed_ns_sink{};
  std::span<u64> worker_tail_wait_ns_sink{};
  std::vector<u32> partitions_per_worker{};
  std::vector<u64> worker_start_offset_ns{};
  std::vector<u64> worker_elapsed_ns{};
  std::vector<u64> worker_tail_wait_ns{};
};

} // namespace rund::kernel
