#pragma once

#include "../../../accel/cpu/simd/dispatch.hpp"

#include <kernel/program/compute/model.hpp>
#include <kernel/program/compute/tile/run.hpp>

#include <rund/counter.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace rund::compute::detail {

class CpuPreparedArena;

// The checked M4 Pro performance profile reports a 128-byte coherence line.
// Map SIMD scratch and Map SIMD counters use this one alignment/stride
// authority. Cache-line isolation on another profile requires a recorded
// power-of-two line width that divides this envelope.
inline constexpr std::size_t kCpuWorkerWriteIsolationBytes = 128u;
static_assert((kCpuWorkerWriteIsolationBytes &
               (kCpuWorkerWriteIsolationBytes - 1u)) == 0u);
static_assert(kCpuWorkerWriteIsolationBytes % alignof(std::max_align_t) == 0u);
static_assert(kCpuWorkerWriteIsolationBytes % sizeof(std::max_align_t) == 0u);

struct alignas(kCpuWorkerWriteIsolationBytes) CpuSimdCount final {
  std::uint64_t vectors{};
  std::uint64_t tails{};
};
static_assert(sizeof(CpuSimdCount) == kCpuWorkerWriteIsolationBytes);

struct CpuProgram final {
  kernel::ComputeMap map;
  node::accel::cpu_simd_detail::CpuSimdDispatch dispatch;
  std::vector<std::uint64_t> input_bytes;
  std::vector<std::uint64_t> input_counts;
  std::vector<kernel::ReadRoute> read_routes;
  kernel::ComputeTileExecutor tile_plan;
  std::size_t scratch_words{};
  std::uint32_t workers{};
  std::uint64_t tile_size{};
};

struct CpuMapRun;
struct CpuMapRoute;

struct CpuMapTileContext final {
  CpuProgram *program = nullptr;
  CpuMapRun *run = nullptr;
  CpuMapRoute *route = nullptr;
  const std::atomic_bool *cancel = nullptr;
};

struct CpuMapRun final {
  kernel::ComputeTileRunPlan tile_plan;
  kernel::ComputeTileExecutor tiles;
  // Pipeline/Program steps are serial. These typed views borrow the active
  // execution arena's component-wise maximum instead of owning one allocation
  // per Map and worker.
  std::span<std::max_align_t> scratch;
  std::span<CpuSimdCount> simd;
  CpuPreparedArena *execution{};
  std::size_t scratch_words{};
  std::size_t workers{};
};

// One immutable Pipeline route owns only resolved Buffer addresses and Views.
// Tile scheduling, worker scratch, and SIMD counters are mutable execution
// storage shared by every serial route of the same Program.
struct CpuMapRoute final {
  node::accel::cpu_simd_detail::CpuSimdBindingView bindings{};
  CpuMapTileContext tile{};
  bool bindings_frozen{};
};

inline void reset_simd(CpuMapRun &run) noexcept {
  for (CpuSimdCount &count : run.simd) {
    count = {};
  }
}

inline void record_simd(CpuMapRun &run, const std::uint32_t worker,
                        const node::accel::CpuSimdRunResult &result) noexcept {
  if (worker >= run.simd.size()) {
    return;
  }
  ::rund::detail::counter::Accumulate(run.simd[worker].vectors,
                                      result.vector_chunk_count);
  ::rund::detail::counter::Accumulate(run.simd[worker].tails,
                                      result.tail_chunk_count);
}

[[nodiscard]] inline CpuSimdCount sum_simd(const CpuMapRun &run) noexcept {
  CpuSimdCount total{};
  for (const CpuSimdCount &count : run.simd) {
    ::rund::detail::counter::Accumulate(total.vectors, count.vectors);
    ::rund::detail::counter::Accumulate(total.tails, count.tails);
  }
  return total;
}

} // namespace rund::compute::detail
