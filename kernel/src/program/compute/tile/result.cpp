#include "state.hpp"

#include <algorithm>
#include <limits>

namespace rund::kernel::compute_tile_detail {
ComputeTileRunResult Project(const ComputeTileRunStorage &storage,
                             const bool ok, const char *const reason,
                             const u32 worker_count, const u32 dispatch_count,
                             const bool report_tail_on_failure) noexcept {
  u32 participating = 0u;
  u32 total = 0u;
  u32 minimum = std::numeric_limits<u32>::max();
  u32 maximum = 0u;
  for (u32 worker = 0u; worker < storage.worker_count_; ++worker) {
    const u32 tiles = storage.worker_tiles_[worker];
    total += tiles;
    if (tiles == 0u) {
      continue;
    }
    ++participating;
    minimum = std::min(minimum, tiles);
    maximum = std::max(maximum, tiles);
  }

  ComputeTileRunResult result{
      .reason = reason != nullptr ? reason : "compute_tile_backend_failed",
      .completed_tile_count =
          storage.completed_.load(std::memory_order_acquire),
      .worker_count = worker_count,
      .participating_workers = participating,
      .worker_tile_count = total,
      .min_tiles_per_worker = participating == 0u ? 0u : minimum,
      .max_tiles_per_worker = maximum,
      .backend_dispatch_count = dispatch_count,
  };
  if (!ok) {
    if (report_tail_on_failure) {
      result.last_tile_units = storage.active_last_tile_units_;
    }
    return result;
  }

  const u32 first = storage.first_failure_.load(std::memory_order_acquire);
  if (first != kNoComputeTileFailure && first < storage.failure_count_) {
    result.reason = storage.failures_[first] != nullptr
                        ? storage.failures_[first]
                        : "compute_tile_callback_failed";
    result.first_failure_tile = first;
  } else {
    result.ok = true;
    result.reason = "pass";
  }
  result.last_tile_units = storage.active_last_tile_units_;
  return result;
}

} // namespace rund::kernel::compute_tile_detail
