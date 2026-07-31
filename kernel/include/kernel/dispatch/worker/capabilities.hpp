#pragma once

#include <kernel/dispatch/worker/enums.hpp>

namespace rund::kernel {

struct WorkerBackendCapabilities {
  u32 backend_width = 0u;
  bool width_matches_request = false;
  bool supports_static_partitions = false;
  bool supports_async_partitions = false;
  bool supports_static_tile_map = false;
  bool supports_claim_free_static_tiles = false;
  bool supports_no_alloc_worker_stats = false;
  // Capability metadata only; strict-FP has no hardware shortcut here.
  bool supports_strict_fp_fold = false;
  bool is_nested = false;
  // Truth only when the backend provider verified the mapping.
  bool affinity_is_truth = false;
  WorkerTruthLevel affinity_truth_level = WorkerTruthLevel::Unknown;
  const u32* worker_capacity_milli = nullptr;
  u32 worker_capacity_count = 0u;
  WorkerTruthLevel worker_capacity_truth_level = WorkerTruthLevel::Unknown;
  WorkerAffinityPolicy affinity_policy = WorkerAffinityPolicy::Static;
};

} // namespace rund::kernel
