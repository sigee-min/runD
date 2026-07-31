#pragma once

#include "../pool.hpp"

namespace kernel_contract_test {

inline rund::kernel::u32 FakeWorkerCount(void* context) {
  return static_cast<FakePool*>(context)->workers;
}

inline rund::kernel::WorkerAffinityPolicy FakeAffinityPolicy(void*) {
  return rund::kernel::WorkerAffinityPolicy::Static;
}

inline bool FakeIsNested(void* context) {
  const FakePool* const pool = static_cast<FakePool*>(context);
  return pool != nullptr && pool->nested;
}

inline rund::kernel::WorkerBackendCapabilities FakeCapabilities(void* context) {
  const FakePool* const pool = static_cast<FakePool*>(context);
  if (pool == nullptr) {
    return {};
  }
  return rund::kernel::WorkerBackendCapabilities{
      .backend_width = pool->workers,
      .width_matches_request = false,
      .supports_static_partitions = true,
      .supports_static_tile_map = true,
      .supports_claim_free_static_tiles = true,
      .supports_no_alloc_worker_stats = pool->supports_no_alloc_stats,
      .supports_strict_fp_fold = pool->supports_strict_fp_fold,
      .is_nested = pool->nested,
      .affinity_is_truth = pool->affinity_truth_level == rund::kernel::WorkerTruthLevel::Verified,
      .affinity_truth_level = pool->affinity_truth_level,
      .affinity_policy = rund::kernel::WorkerAffinityPolicy::Static,
  };
}

} // namespace kernel_contract_test
