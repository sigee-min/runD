#include "local.hpp"

namespace rund::kernel::dispatch::detail {
namespace {

u32 WorkerCount(const WorkerBackend& backend) {
  return backend.worker_count != nullptr ? backend.worker_count(backend.context) : 0u;
}

WorkerAffinityPolicy AffinityPolicy(const WorkerBackend& backend) {
  return backend.affinity_policy != nullptr ? backend.affinity_policy(backend.context)
                                            : WorkerAffinityPolicy::Static;
}

bool IsNested(const WorkerBackend& backend) {
  return backend.is_nested != nullptr && backend.is_nested(backend.context);
}

} // namespace

WorkerBackendCapabilities InspectBackend(const WorkerBackend& backend,
                                         const u32 requested_width) {
  if (!backend) {
    return WorkerBackendCapabilities{};
  }
  WorkerBackendCapabilities capabilities =
      backend.capabilities != nullptr ? backend.capabilities(backend.context) : WorkerBackendCapabilities{};
  if (capabilities.backend_width == 0u) {
    capabilities.backend_width = WorkerCount(backend);
  }
  capabilities.width_matches_request = capabilities.backend_width == requested_width;
  capabilities.affinity_policy = AffinityPolicy(backend);
  capabilities.is_nested = IsNested(backend);
  if (capabilities.supports_static_partitions) {
    capabilities.supports_static_tile_map = true;
    capabilities.supports_claim_free_static_tiles = true;
  }
  capabilities.supports_async_partitions =
      capabilities.supports_async_partitions &&
      backend.submit_partitions != nullptr;
  if (capabilities.affinity_truth_level == WorkerTruthLevel::Unknown && capabilities.affinity_is_truth) {
    capabilities.affinity_truth_level = WorkerTruthLevel::Verified;
  }
  if (backend.capabilities == nullptr) {
    capabilities.supports_static_partitions = backend.execute_partitions != nullptr;
    capabilities.supports_async_partitions =
        backend.submit_partitions != nullptr;
    capabilities.supports_static_tile_map = capabilities.supports_static_partitions;
    capabilities.supports_claim_free_static_tiles = capabilities.supports_static_tile_map;
    capabilities.supports_no_alloc_worker_stats = false;
    capabilities.supports_strict_fp_fold = false;
    capabilities.affinity_is_truth = false;
    capabilities.affinity_truth_level =
        backend.affinity_policy != nullptr ? WorkerTruthLevel::HintOnly : WorkerTruthLevel::Unknown;
  }
  return capabilities;
}

bool ExecuteBackendPartitions(const WorkerBackend& backend,
                              const Partition* partitions,
                              const u32 partition_count,
                              const WorkerTask task,
                              WorkerStats* const out_stats) {
  return backend.execute_partitions != nullptr &&
         backend.execute_partitions(backend.context, partitions, partition_count, task, out_stats);
}

} // namespace rund::kernel::dispatch::detail
