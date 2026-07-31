#include "local.hpp"

namespace program_skeleton_contract {

int test_skeleton_scheduled_executor_capacity() {
  using namespace kernel_contract_test;

  std::vector<rund::kernel::i32> scheduled_storage(8192u);
  auto scheduled_view = rund::kernel::view<rund::kernel::i32>(
      scheduled_storage.data(),
      rund::kernel::Index<1u>{scheduled_storage.size()});
  TEST_ASSERT(scheduled_view);
  rund::kernel::Workspace scheduled_workspace{};
  FakePool scheduled_pool = BuildStaticPool(4u);
  const rund::kernel::Executor scheduled_exec = rund::kernel::executor(
      scheduled_workspace, MakeFakeBackend(&scheduled_pool), 4u,
      rund::kernel::align(4u));
  TEST_ASSERT(scheduled_exec.valid);
  const rund::kernel::SkeletonResult scheduled_result = rund::kernel::each(
      scheduled_exec, rund::kernel::space(scheduled_storage.size()),
      [&](auto index) noexcept {
        scheduled_view(index) = rund::math32::detail::ScalarAddWrap(
            static_cast<rund::kernel::i32>(index[0]), 11);
      });
  TEST_ASSERT(scheduled_result.ok);
  TEST_ASSERT(scheduled_result.visited_count == scheduled_storage.size());
  TEST_ASSERT(scheduled_result.partition_boundary_checked);
  TEST_ASSERT(scheduled_result.partition_boundary_aligned);
  TEST_ASSERT(scheduled_result.boundary_alignment_units == 4u);
  TEST_ASSERT(scheduled_workspace.program.ok);
  TEST_ASSERT(scheduled_workspace.program.schedule.execution_width == 4u);
  TEST_ASSERT(scheduled_workspace.program.schedule.partition_count == 4u);
  TEST_ASSERT(scheduled_workspace.program.exec_kernel.physical_tiling_enabled);
  TEST_ASSERT(scheduled_workspace.program.exec_kernel.physical_tile_units ==
              684u);
  TEST_ASSERT(scheduled_workspace.program.exec_kernel.physical_tile_count ==
              12u);
  for (std::size_t index = 0u; index < scheduled_storage.size(); ++index) {
    TEST_ASSERT(scheduled_storage[index] ==
                rund::math32::detail::ScalarAddWrap(
                    static_cast<rund::kernel::i32>(index), 11));
  }
  const rund::kernel::PreparedEach<1u> prepared_each =
      scheduled_exec.prepare(rund::kernel::space(scheduled_storage.size()));
  TEST_ASSERT(prepared_each);
  TEST_ASSERT(scheduled_workspace.program.ok);
  TEST_ASSERT(scheduled_workspace.program.schedule.execution_width == 4u);
  TEST_ASSERT(scheduled_workspace.program.schedule.partition_count == 4u);
  TEST_ASSERT(prepared_each.physical_tiling_enabled);
  TEST_ASSERT(prepared_each.physical_tile_units == 684u);
  TEST_ASSERT(prepared_each.physical_tile_count == 12u);
  const rund::kernel::SkeletonResult prepared_first =
      prepared_each.run([&](auto index) noexcept {
        scheduled_view(index) = rund::math32::detail::ScalarAddWrap(
            static_cast<rund::kernel::i32>(index[0]), 21);
      });
  TEST_ASSERT(prepared_first.ok);
  TEST_ASSERT(prepared_first.visited_count == scheduled_storage.size());
  TEST_ASSERT(prepared_first.partition_boundary_checked);
  TEST_ASSERT(prepared_first.partition_boundary_aligned);
  TEST_ASSERT(prepared_first.boundary_alignment_units == 4u);
  TEST_ASSERT(prepared_first.physical_tile_units == 684u);
  const rund::kernel::SkeletonResult prepared_second =
      prepared_each.run([&](auto index) noexcept {
        scheduled_view(index) =
            rund::math32::detail::ScalarAddWrap(scheduled_view(index), 5);
      });
  TEST_ASSERT(prepared_second.ok);
  TEST_ASSERT(prepared_second.visited_count == scheduled_storage.size());
  for (std::size_t index = 0u; index < scheduled_storage.size(); ++index) {
    TEST_ASSERT(scheduled_storage[index] ==
                rund::math32::detail::ScalarAddWrap(
                    static_cast<rund::kernel::i32>(index), 26));
  }
  const rund::kernel::PreparedEach<1u> replacement_prepared =
      scheduled_exec.prepare(rund::kernel::space(8u));
  TEST_ASSERT(replacement_prepared);
  TEST_ASSERT(scheduled_workspace.program.schedule.execution_width == 4u);
  TEST_ASSERT(scheduled_workspace.program.schedule.partition_count == 4u);
  TEST_ASSERT(replacement_prepared.physical_tiling_enabled);
  TEST_ASSERT(replacement_prepared.physical_tile_units == 1u);
  TEST_ASSERT(replacement_prepared.physical_tile_count == 8u);
  TEST_ASSERT(ExpectReason("prepared_each_program_mismatch",
                           prepared_each.run([](auto) noexcept {})) == 0);
  const rund::kernel::PreparedEach<1u> same_shape_prepared =
      scheduled_exec.prepare(rund::kernel::space(8u));
  TEST_ASSERT(same_shape_prepared);
  TEST_ASSERT(ExpectReason("prepared_each_program_mismatch",
                           replacement_prepared.run([](auto) noexcept {})) ==
              0);
  return 0;
}

} // namespace program_skeleton_contract
