#include "local.hpp"

#include <array>
#include <string_view>

namespace program_compute_contract::tile_contract {

int CheckConstCallback() {
  auto pool = kernel_contract_test::BuildStaticPool(1u);
  ComputeTileExecutor executor{kernel_contract_test::MakeFakeBackend(&pool),
                               1u};
  TEST_ASSERT(executor.prepare(4u).ok);
  ComputeTileExecutor run = executor.make_run();
  TEST_ASSERT(run.prepared());
  const auto callback = [](const ComputeTile &) {
    return ComputeTileCallbackResult{};
  };
  TEST_ASSERT(run.run(callback).ok);
  return 0;
}

int CheckIndependentRunState() {
  auto pool = kernel_contract_test::BuildStaticPool(Workers);
  ComputeTileExecutor plan =
      MakeExecutor(kernel_contract_test::MakeFakeBackend(&pool), Workers);
  TEST_ASSERT(plan.prepare(Boundary + 1u).ok);
  ComputeTileExecutor first = plan.make_run();
  ComputeTileExecutor second = plan.make_run();
  TEST_ASSERT(first.prepared());
  TEST_ASSERT(second.prepared());
  std::array<u32, Boundary + 1u> first_values{};
  std::array<u32, Boundary + 1u> second_values{};
  TEST_ASSERT(first
                  .run([&](const ComputeTile &tile) {
                    for (u32 index = tile.begin; index < tile.end; ++index) {
                      first_values[index] = index + 1u;
                    }
                    return tile.index == 1u
                               ? ComputeTileCallbackResult{false,
                                                           "first_run_failed"}
                               : ComputeTileCallbackResult{};
                  })
                  .reason == std::string_view{"first_run_failed"});
  const auto second_result = second.run([&](const ComputeTile &tile) {
    for (u32 index = tile.begin; index < tile.end; ++index) {
      second_values[index] = index + 2u;
    }
    return ComputeTileCallbackResult{};
  });
  TEST_ASSERT(second_result.ok);
  TEST_ASSERT(second_result.completed_tile_count == plan.tile_count());
  for (u32 index = 0u; index < Boundary + 1u; ++index) {
    TEST_ASSERT(second_values[index] == index + 2u);
  }
  return 0;
}

} // namespace program_compute_contract::tile_contract
