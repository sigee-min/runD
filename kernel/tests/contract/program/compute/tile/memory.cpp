#include "local.hpp"

#include <limits>
#include <string_view>
#include <utility>

namespace program_compute_contract::tile_contract {

int CheckRetainedMemory() {
  ComputeTileExecutor empty{};
  const ComputeTileRetainedMemory empty_memory = empty.retained_memory();
  const auto empty_run_memory = empty.run_plan().storage_plan().retained;
  TEST_ASSERT(empty_memory.state_bytes == 0u);
  TEST_ASSERT(empty_memory.workspace_bytes == 0u);
  TEST_ASSERT(empty_memory.failure_slot_bytes == 0u);
  TEST_ASSERT(empty_memory.worker_tile_bytes == 0u);
  TEST_ASSERT(empty_memory.async_context_bytes == 0u);
  TEST_ASSERT(empty_memory.total_bytes == 0u);
  TEST_ASSERT(!empty_run_memory.ok);
  TEST_ASSERT(std::string_view{empty_run_memory.reason} ==
              "compute_tile_run_memory_not_prepared");
  TEST_ASSERT(empty_run_memory.memory.total_bytes == 0u);

  constexpr auto overflow = PlanComputeTileRunMemory(ComputeTileRetainedMemory{
      .state_bytes = std::numeric_limits<u64>::max(),
      .workspace_bytes = 1u,
  });
  static_assert(!overflow.ok);
  static_assert(overflow.memory.total_bytes == std::numeric_limits<u64>::max());
  TEST_ASSERT(std::string_view{overflow.reason} ==
              "compute_tile_run_memory_overflow");

  auto pool = kernel_contract_test::BuildStaticPool(Workers);
  ComputeTileExecutor plan =
      MakeExecutor(kernel_contract_test::MakeFakeBackend(&pool), Workers);
  const ComputeTileRetainedMemory constructed = plan.retained_memory();
  const auto unprepared_run_memory = plan.run_plan().storage_plan().retained;
  TEST_ASSERT(constructed.state_bytes == 0u);
  TEST_ASSERT(constructed.total_bytes == TotalBytes(constructed));
  TEST_ASSERT(!unprepared_run_memory.ok);
  TEST_ASSERT(std::string_view{unprepared_run_memory.reason} ==
              "compute_tile_run_memory_not_prepared");
  const auto prepared = plan.prepare(Boundary + 1u);
  TEST_ASSERT(prepared.ok);
  const ComputeTileRetainedMemory planned = plan.retained_memory();
  TEST_ASSERT(planned.state_bytes != 0u);
  TEST_ASSERT(planned.workspace_bytes != 0u);
  TEST_ASSERT(planned.failure_slot_bytes == 0u);
  TEST_ASSERT(planned.worker_tile_bytes == 0u);
  TEST_ASSERT(planned.async_context_bytes == 0u);
  TEST_ASSERT(planned.total_bytes == TotalBytes(planned));

  kernel_contract_test::memory_allocation::Reset();
  const auto first_run_plan = plan.run_plan().storage_plan().retained;
  const auto second_run_plan = plan.run_plan().storage_plan().retained;
  kernel_contract_test::memory_allocation::Stop();
  TEST_ASSERT(kernel_contract_test::memory_allocation::Count() == 0u);
  TEST_ASSERT(first_run_plan.ok);
  TEST_ASSERT(second_run_plan.ok);
  TEST_ASSERT(std::string_view{first_run_plan.reason} == "pass");
  TEST_ASSERT(SameMemory(first_run_plan.memory, second_run_plan.memory));
  TEST_ASSERT(first_run_plan.memory.state_bytes != 0u);
  TEST_ASSERT(first_run_plan.memory.workspace_bytes ==
              static_cast<u64>(prepared.worker_count) *
                  (sizeof(u32) + 3u * sizeof(u64)));
  TEST_ASSERT(first_run_plan.memory.failure_slot_bytes ==
              static_cast<u64>(prepared.tile_count) * sizeof(const char *));
  TEST_ASSERT(first_run_plan.memory.worker_tile_bytes ==
              static_cast<u64>(prepared.worker_count) * sizeof(u32));
  TEST_ASSERT(first_run_plan.memory.async_context_bytes == 0u);

  kernel_contract_test::memory_allocation::Reset();
  ComputeTileExecutor run = plan.make_run();
  kernel_contract_test::memory_allocation::Stop();
  TEST_ASSERT(run.prepared());
  TEST_ASSERT(kernel_contract_test::memory_allocation::Count() == 7u);
  const ComputeTileRetainedMemory retained = run.retained_memory();
  TEST_ASSERT(SameMemory(retained, first_run_plan.memory));
  TEST_ASSERT(retained.total_bytes == TotalBytes(retained));

  kernel_contract_test::memory_allocation::Reset();
  const ComputeTileRetainedMemory first = plan.retained_memory();
  const ComputeTileRetainedMemory second = run.retained_memory();
  kernel_contract_test::memory_allocation::Stop();
  TEST_ASSERT(kernel_contract_test::memory_allocation::Count() == 0u);
  TEST_ASSERT(SameMemory(first, planned));
  TEST_ASSERT(SameMemory(second, retained));

  kernel_contract_test::memory_allocation::Reset();
  const auto hot =
      run.run([](const ComputeTile &) { return ComputeTileCallbackResult{}; });
  kernel_contract_test::memory_allocation::Stop();
  TEST_ASSERT(hot.ok);
  TEST_ASSERT(kernel_contract_test::memory_allocation::Count() == 0u);
  TEST_ASSERT(SameMemory(run.retained_memory(), retained));

  ComputeTileExecutor moved = std::move(run);
  TEST_ASSERT(SameMemory(moved.retained_memory(), retained));
  TEST_ASSERT(run.retained_memory().total_bytes == 0u);
  return 0;
}

} // namespace program_compute_contract::tile_contract
