#include "local.hpp"

#include <array>
#include <atomic>
#include <string_view>

namespace program_compute_contract::tile_contract {

int CheckBoundedBind() {
  constexpr u32 active_count = Boundary - 1u;
  constexpr u32 active_tiles = 16u;
  constexpr u32 active_tail = TileUnits - 1u;
  auto pool = kernel_contract_test::BuildStaticPool(Workers);
  ComputeTileExecutor source =
      MakeExecutor(kernel_contract_test::MakeFakeBackend(&pool), Workers);
  TEST_ASSERT(source.prepare(Boundary + 1u).ok);
  const auto plan = source.run_plan();
  ExternalRunStorage<17u, Workers> external{};

  kernel_contract_test::memory_allocation::Reset();
  ComputeTileExecutor bounded = plan.bind(external.view(), active_count);
  kernel_contract_test::memory_allocation::Stop();
  TEST_ASSERT(kernel_contract_test::memory_allocation::Count() == 0u);
  TEST_ASSERT(bounded.prepared());
  TEST_ASSERT(bounded.count() == active_count);
  TEST_ASSERT(bounded.tile_count() == active_tiles);
  TEST_ASSERT(plan.count() == Boundary + 1u);
  TEST_ASSERT(plan.tile_count() == 17u);

  std::array<std::atomic<u32>, Boundary + 2u> visits{};
  std::atomic<bool> invalid_tile{false};
  kernel_contract_test::memory_allocation::Reset();
  const auto result = bounded.run([&](const ComputeTile &tile) {
    if (tile.begin >= tile.end || tile.end > active_count) {
      invalid_tile.store(true, std::memory_order_relaxed);
      return ComputeTileCallbackResult{.ok = false,
                                       .reason = "test_invalid_tile"};
    }
    for (u32 index = tile.begin; index < tile.end; ++index) {
      visits[index].fetch_add(1u, std::memory_order_relaxed);
    }
    return ComputeTileCallbackResult{};
  });
  kernel_contract_test::memory_allocation::Stop();
  TEST_ASSERT(result.ok);
  TEST_ASSERT(!invalid_tile.load(std::memory_order_relaxed));
  TEST_ASSERT(result.completed_tile_count == active_tiles);
  TEST_ASSERT(result.worker_tile_count == active_tiles);
  TEST_ASSERT(result.last_tile_units == active_tail);
  TEST_ASSERT(kernel_contract_test::memory_allocation::Count() == 0u);
  for (u32 index = 0u; index < active_count; ++index) {
    TEST_ASSERT(visits[index].load(std::memory_order_relaxed) == 1u);
  }
  for (u32 index = active_count; index < visits.size(); ++index) {
    TEST_ASSERT(visits[index].load(std::memory_order_relaxed) == 0u);
  }

  ComputeTileExecutor empty = plan.bind(external.view(), 0u);
  TEST_ASSERT(empty.prepared());
  TEST_ASSERT(empty.count() == 0u);
  TEST_ASSERT(empty.tile_count() == 0u);
  TEST_ASSERT(empty
                  .run([](const ComputeTile &) {
                    return ComputeTileCallbackResult{.ok = false,
                                                     .reason = "must_not_run"};
                  })
                  .ok);

  ComputeTileExecutor invalid = plan.bind(external.view(), Boundary + 2u);
  TEST_ASSERT(!invalid.prepared());
  const auto rejected = invalid.run(
      [](const ComputeTile &) { return ComputeTileCallbackResult{}; });
  TEST_ASSERT(std::string_view{rejected.reason} ==
              "compute_tile_run_active_count_invalid");

  ComputeTileExecutor untiled_source{
      kernel_contract_test::MakeFakeBackend(&pool), Workers,
      rund::kernel::no_physical_tiles()};
  TEST_ASSERT(untiled_source.prepare(Boundary + 1u).ok);
  const auto untiled_plan = untiled_source.run_plan();
  TEST_ASSERT(untiled_plan.tile_count() == Workers);
  ExternalRunStorage<Workers, Workers> untiled_external{};
  ComputeTileExecutor untiled = untiled_plan.bind(untiled_external.view(), 1u);
  TEST_ASSERT(untiled.prepared());
  TEST_ASSERT(untiled.count() == 1u);
  TEST_ASSERT(untiled.tile_count() == 1u);
  u32 untiled_visits = 0u;
  const auto untiled_result = untiled.run([&](const ComputeTile &tile) {
    if (tile.index != 0u || tile.begin != 0u || tile.end != 1u) {
      return ComputeTileCallbackResult{.ok = false,
                                       .reason = "test_invalid_tile"};
    }
    ++untiled_visits;
    return ComputeTileCallbackResult{};
  });
  TEST_ASSERT(untiled_result.ok);
  TEST_ASSERT(untiled_result.completed_tile_count == 1u);
  TEST_ASSERT(untiled_result.last_tile_units == 1u);
  TEST_ASSERT(untiled_visits == 1u);
  return 0;
}

} // namespace program_compute_contract::tile_contract
