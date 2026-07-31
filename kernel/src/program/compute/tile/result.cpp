#include "state.hpp"

#include <algorithm>
#include <limits>

namespace rund::kernel::compute_tile_detail {
namespace {

[[nodiscard]] u32 Tail(const u32 count, const u32 tile_count,
                       const u32 tile_units) noexcept {
  if (tile_count == 0u || tile_units == 0u) {
    return 0u;
  }
  return count - (tile_count - 1u) * tile_units;
}

} // namespace

ComputeTileRunResult Project(const Context &context,
                             const Completion &completion, const u32 count,
                             const u32 tile_count,
                             const u32 tile_units) noexcept {
  u32 participating = 0u;
  u32 total = 0u;
  u32 minimum = std::numeric_limits<u32>::max();
  u32 maximum = 0u;
  for (u32 worker = 0u; worker < context.worker_count; ++worker) {
    const u32 tiles = context.worker_tiles[worker];
    total += tiles;
    if (tiles == 0u) {
      continue;
    }
    ++participating;
    minimum = std::min(minimum, tiles);
    maximum = std::max(maximum, tiles);
  }

  ComputeTileRunResult result{
      .reason = completion.reason,
      .completed_tile_count = context.completed.load(std::memory_order_acquire),
      .worker_count = completion.worker_count,
      .participating_workers = participating,
      .worker_tile_count = total,
      .min_tiles_per_worker = participating == 0u ? 0u : minimum,
      .max_tiles_per_worker = maximum,
      .backend_dispatch_count = completion.dispatch_count,
  };
  if (!completion.ok) {
    if (completion.report_tail_on_failure) {
      result.last_tile_units = Tail(count, tile_count, tile_units);
    }
    return result;
  }

  const u32 first = context.first_failure.load(std::memory_order_acquire);
  if (first != kNoComputeTileFailure && first < context.failure_count) {
    result.reason = context.failures[first] != nullptr
                        ? context.failures[first]
                        : "compute_tile_callback_failed";
    result.first_failure_tile = first;
  } else {
    result.ok = true;
    result.reason = "pass";
  }
  result.last_tile_units = Tail(count, tile_count, tile_units);
  return result;
}

} // namespace rund::kernel::compute_tile_detail
