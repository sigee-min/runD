#include "local.hpp"

namespace program_skeleton_contract {

int test_skeleton_explicit_physical_tile_policy_report() {
  using namespace kernel_contract_test;

  std::vector<rund::kernel::i32> explicit_tile_storage(1000u);
  auto explicit_tile_view = rund::kernel::view<rund::kernel::i32>(
      explicit_tile_storage.data(),
      rund::kernel::Index<1u>{explicit_tile_storage.size()});
  TEST_ASSERT(explicit_tile_view);
  rund::kernel::Workspace explicit_tile_workspace{};
  FakePool explicit_tile_pool = BuildStaticPool(4u);
  const rund::kernel::Executor explicit_tile_exec = rund::kernel::executor(
      explicit_tile_workspace, MakeFakeBackend(&explicit_tile_pool), 4u,
      rund::kernel::align(1u), rund::kernel::physical_tiles_per_worker(5u));
  TEST_ASSERT(explicit_tile_exec.valid);
  const rund::kernel::PreparedEach<1u> explicit_tile_prepared =
      explicit_tile_exec.prepare(
          rund::kernel::space(explicit_tile_storage.size()));
  TEST_ASSERT(explicit_tile_prepared);
  TEST_ASSERT(explicit_tile_workspace.program.ok);
  TEST_ASSERT(explicit_tile_workspace.program.schedule.execution_width == 4u);
  TEST_ASSERT(explicit_tile_workspace.program.schedule.partition_count == 4u);
  TEST_ASSERT(explicit_tile_prepared.physical_tiling_enabled);
  TEST_ASSERT(explicit_tile_prepared.physical_tile_units == 50u);
  TEST_ASSERT(explicit_tile_prepared.physical_tile_count == 20u);
  TEST_ASSERT(explicit_tile_workspace.program.exec_kernel.tile_count == 20u);
  const rund::kernel::SkeletonResult explicit_tile_result =
      explicit_tile_prepared.run([&](auto index) noexcept {
        explicit_tile_view(index) = static_cast<rund::kernel::i32>(index[0]);
      });
  TEST_ASSERT(explicit_tile_result.ok);
  TEST_ASSERT(explicit_tile_result.visited_count ==
              explicit_tile_storage.size());
  TEST_ASSERT(explicit_tile_result.physical_tile_units == 50u);
  const rund::kernel::KernelExecutionReport explicit_tile_report =
      rund::kernel::execution_report(explicit_tile_exec);
  TEST_ASSERT(explicit_tile_report.observed);
  TEST_ASSERT(explicit_tile_report.ok);
  TEST_ASSERT(explicit_tile_report.kind ==
              rund::kernel::KernelExecutionReportKind::Narrow);
  TEST_ASSERT(explicit_tile_report.packet_count ==
              explicit_tile_storage.size());
  TEST_ASSERT(explicit_tile_report.execution_width == 4u);
  TEST_ASSERT(explicit_tile_report.partition_count == 4u);
  TEST_ASSERT(explicit_tile_report.physical_tiling_enabled);
  TEST_ASSERT(explicit_tile_report.physical_tile_units == 50u);
  TEST_ASSERT(explicit_tile_report.physical_tile_count == 20u);
  TEST_ASSERT(explicit_tile_report.worker_tile_count == 20u);
  TEST_ASSERT(explicit_tile_report.min_tiles_per_worker == 5u);
  TEST_ASSERT(explicit_tile_report.max_tiles_per_worker == 5u);
  TEST_ASSERT(explicit_tile_report.tile_imbalance_milli == 0u);
  TEST_ASSERT(explicit_tile_report.dispatch_cost_measured);
  TEST_ASSERT(explicit_tile_report.telemetry_update_cost_measured);
  for (std::size_t index = 0u; index < explicit_tile_storage.size(); ++index) {
    TEST_ASSERT(explicit_tile_storage[index] ==
                static_cast<rund::kernel::i32>(index));
  }
  return 0;
}

} // namespace program_skeleton_contract
