#include "local.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>

namespace program_compute_contract::tile_contract {
namespace {

int CheckBoundary(const u32 count, const u32 expected_tiles,
                  const u32 expected_tail) {
  auto pool = kernel_contract_test::BuildStaticPool(Workers);
  ComputeTileExecutor executor =
      MakeExecutor(kernel_contract_test::MakeFakeBackend(&pool), Workers);
  const auto prepared = executor.prepare(count);
  if (!prepared.ok) {
    std::fprintf(stderr, "compute tile prepare failed: %s\n", prepared.reason);
  }
  TEST_ASSERT(prepared.ok);
  TEST_ASSERT(prepared.tile_units == TileUnits);
  TEST_ASSERT(prepared.tile_count == expected_tiles);
  ComputeTileExecutor run_executor = executor.make_run();
  TEST_ASSERT(run_executor.prepared());

  std::array<std::atomic<u32>, Boundary + 2u> visits{};
  std::atomic<bool> invalid_tile{false};
  const auto run = run_executor.run([&](const ComputeTile &tile) {
    if (tile.begin >= tile.end || tile.end > count) {
      invalid_tile.store(true, std::memory_order_relaxed);
      return ComputeTileCallbackResult{
          .ok = false,
          .reason = "test_invalid_tile",
      };
    }
    for (u32 index = tile.begin; index < tile.end; ++index) {
      visits[index].fetch_add(1u, std::memory_order_relaxed);
    }
    return ComputeTileCallbackResult{};
  });
  if (!run.ok) {
    std::fprintf(stderr, "compute tile boundary run failed: %s\n", run.reason);
  }
  TEST_ASSERT(run.ok);
  TEST_ASSERT(!invalid_tile.load(std::memory_order_relaxed));
  TEST_ASSERT(run.completed_tile_count == expected_tiles);
  TEST_ASSERT(run.worker_tile_count == expected_tiles);
  TEST_ASSERT(run.backend_dispatch_count == 1u);
  TEST_ASSERT(run.worker_count == Workers);
  TEST_ASSERT(run.participating_workers >= 2u);
  for (u32 index = 0u; index < count; ++index) {
    TEST_ASSERT(visits[index].load(std::memory_order_relaxed) == 1u);
  }
  TEST_ASSERT(run.last_tile_units == expected_tail);
  return 0;
}

int CheckThresholdFreeDispatch(const u32 count) {
  auto pool = kernel_contract_test::BuildStaticPool(Workers);
  ComputeTileExecutor executor =
      MakeExecutor(kernel_contract_test::MakeFakeBackend(&pool), Workers);
  const auto prepared = executor.prepare(count);
  TEST_ASSERT(prepared.ok);
  TEST_ASSERT(prepared.worker_count == Workers);
  TEST_ASSERT(prepared.tile_count != 0u);
  ComputeTileExecutor run_executor = executor.make_run();
  TEST_ASSERT(run_executor.prepared());

  std::array<std::atomic<u32>, Boundary + 2u> visits{};
  std::atomic<u32> callbacks{0u};
  const auto run = run_executor.run([&](const ComputeTile &tile) {
    callbacks.fetch_add(1u, std::memory_order_relaxed);
    for (u32 index = tile.begin; index < tile.end; ++index) {
      visits[index].fetch_add(1u, std::memory_order_relaxed);
    }
    return ComputeTileCallbackResult{};
  });
  TEST_ASSERT(run.ok);
  TEST_ASSERT(run.worker_count == Workers);
  TEST_ASSERT(run.backend_dispatch_count == 1u);
  TEST_ASSERT(run.worker_tile_count == prepared.tile_count);
  TEST_ASSERT(run.completed_tile_count == prepared.tile_count);
  TEST_ASSERT(callbacks.load(std::memory_order_relaxed) == prepared.tile_count);
  TEST_ASSERT(run.participating_workers ==
              std::min(Workers, prepared.tile_count));
  for (u32 index = 0u; index < count; ++index) {
    TEST_ASSERT(visits[index].load(std::memory_order_relaxed) == 1u);
  }
  return 0;
}

} // namespace

int CheckPositiveDispatch() {
  constexpr std::array counts{
      1u, TileUnits - 1u, TileUnits, TileUnits + 1u, Boundary + 1u,
  };
  for (const u32 count : counts) {
    if (CheckThresholdFreeDispatch(count) != 0) {
      return 1;
    }
  }
  return 0;
}

int CheckBoundaries() {
  if (CheckBoundary(Boundary - 1u, 16u, TileUnits - 1u) != 0) {
    return 1;
  }
  if (CheckBoundary(Boundary, 16u, TileUnits) != 0) {
    return 1;
  }
  return CheckBoundary(Boundary + 1u, 17u, 1u);
}

} // namespace program_compute_contract::tile_contract
