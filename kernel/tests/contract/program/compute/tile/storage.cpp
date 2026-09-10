#include "local.hpp"

#include <string_view>

namespace program_compute_contract::tile_contract {

int CheckBorrowedStorage() {
  auto pool = kernel_contract_test::BuildStaticPool(Workers);
  ComputeTileExecutor source =
      MakeExecutor(kernel_contract_test::MakeFakeBackend(&pool), Workers);
  TEST_ASSERT(source.prepare(Boundary + 1u).ok);
  const auto first_plan = source.run_plan();
  TEST_ASSERT(first_plan.prepared());
  TEST_ASSERT(first_plan.count() == Boundary + 1u);
  TEST_ASSERT(first_plan.tile_count() == 17u);

  const auto direct = source.run(
      [](const ComputeTile &) { return ComputeTileCallbackResult{}; });
  TEST_ASSERT(!direct.ok);
  TEST_ASSERT(std::string_view{direct.reason} ==
              "compute_tile_run_storage_missing");

  ExternalRunStorage<17u, Workers> external{};
  kernel_contract_test::memory_allocation::Reset();
  ComputeTileExecutor first = first_plan.bind(external.view());
  kernel_contract_test::memory_allocation::Stop();
  TEST_ASSERT(kernel_contract_test::memory_allocation::Count() == 0u);
  TEST_ASSERT(first.prepared());
  TEST_ASSERT(first.has_run_storage());
  TEST_ASSERT(first.borrowed_run_storage());
  TEST_ASSERT(!external.state.busy());
  const u64 first_generation = external.state.generation();
  TEST_ASSERT(first_generation != 0u);

  kernel_contract_test::memory_allocation::Reset();
  const auto first_result = first.run(
      [](const ComputeTile &) { return ComputeTileCallbackResult{}; });
  kernel_contract_test::memory_allocation::Stop();
  TEST_ASSERT(first_result.ok);
  TEST_ASSERT(kernel_contract_test::memory_allocation::Count() == 0u);

  TEST_ASSERT(source.prepare(TileUnits + 1u).ok);
  const auto second_plan = source.run_plan();
  TEST_ASSERT(first_plan.count() == Boundary + 1u);
  TEST_ASSERT(second_plan.count() == TileUnits + 1u);
  const auto envelope = rund::kernel::MergeComputeTileRunStoragePlans(
      first_plan.storage_plan(), second_plan.storage_plan());
  TEST_ASSERT(envelope.ok);
  TEST_ASSERT(envelope.failure_slot_capacity == 17u);
  TEST_ASSERT(envelope.worker_capacity == Workers);

  ComputeTileExecutor second = second_plan.bind(external.view());
  TEST_ASSERT(second.prepared());
  TEST_ASSERT(external.state.generation() == first_generation + 1u);
  TEST_ASSERT(!first.prepared());
  const auto rebound = first.run(
      [](const ComputeTile &) { return ComputeTileCallbackResult{}; });
  TEST_ASSERT(!rebound.ok);
  TEST_ASSERT(std::string_view{rebound.reason} == "compute_tile_run_rebound");
  TEST_ASSERT(
      second
          .run([](const ComputeTile &) { return ComputeTileCallbackResult{}; })
          .ok);

  ExternalRunStorage<16u, Workers> undersized{};
  ComputeTileExecutor rejected = first_plan.bind(undersized.view());
  TEST_ASSERT(!rejected.prepared());
  const auto capacity = rejected.run(
      [](const ComputeTile &) { return ComputeTileCallbackResult{}; });
  TEST_ASSERT(std::string_view{capacity.reason} ==
              "compute_tile_run_storage_capacity");
  return 0;
}

} // namespace program_compute_contract::tile_contract
