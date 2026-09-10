#pragma once

#include <kernel/program/compute/tile/run.hpp>

#include <atomic>
#include <cstdint>
#include <span>

namespace rund::compute::detail {

class CpuPreparedArena;

struct CpuCollective final {
  kernel::ComputeTileExecutor tile_plan;
  std::uint64_t tile_size{};
  std::uint32_t tile_count{};
  bool needs_prefixes{};
};

// A collective is bounded by kernel::u32 elements. The largest magnitude is
// therefore below 2^96 for every supported 32/64-bit integer or fixed lane,
// so a signed 128-bit carrier preserves the mathematical sum until the one
// observable storage boundary.
using CpuCollectiveWide = __int128_t;

struct CpuCollectiveRun final {
  kernel::ComputeTileRunPlan tile_plan;
  kernel::ComputeTileExecutor tiles;
  std::span<CpuCollectiveWide> total_capacity;
  std::span<CpuCollectiveWide> prefix_capacity;
  std::span<CpuCollectiveWide> totals;
  std::span<CpuCollectiveWide> prefixes;
  CpuPreparedArena *execution{};
  std::uint64_t tile_size{};
  bool needs_prefixes{};
};

} // namespace rund::compute::detail
