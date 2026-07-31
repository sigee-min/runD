#include "local.hpp"

namespace program_skeleton_contract {

int SkeletonShapeCallbacks() {
  std::array<rund::kernel::Index<2u>, 3u> partition_indices{};
  std::size_t partition_index_count = 0u;
  const rund::kernel::Partition rank2_partition{
      .worker_index = 0u,
      .begin = 2u,
      .end = 5u,
  };
  const rund::kernel::SkeletonResult rank2_partition_result =
      rund::kernel::each(rund::kernel::space(2u, 3u), rank2_partition,
                         [&](auto ij) noexcept {
                           partition_indices[partition_index_count] = ij;
                           ++partition_index_count;
                         });
  TEST_ASSERT(rank2_partition_result.ok);
  TEST_ASSERT(rank2_partition_result.visited_count == 3u);
  TEST_ASSERT(partition_index_count == 3u);
  TEST_ASSERT(partition_indices[0u][0u] == 0u &&
              partition_indices[0u][1u] == 2u);
  TEST_ASSERT(partition_indices[1u][0u] == 1u &&
              partition_indices[1u][1u] == 0u);
  TEST_ASSERT(partition_indices[2u][0u] == 1u &&
              partition_indices[2u][1u] == 1u);

  rund::kernel::u64 rvalue_only_sum = 0u;
  const rund::kernel::SkeletonResult rvalue_only_result =
      rund::kernel::each(rund::kernel::space(2u, 2u),
                         RvalueOnlyIndexCallback{.sum = &rvalue_only_sum});
  TEST_ASSERT(rvalue_only_result.ok);
  TEST_ASSERT(rvalue_only_sum == 22u);

  bool lvalue_callback_called = false;
  bool rvalue_callback_called = false;
  const rund::kernel::SkeletonResult overload_result = rund::kernel::each(
      rund::kernel::space(1u, 2u), OverloadSensitiveIndexCallback{
                                       .lvalue_called = &lvalue_callback_called,
                                       .rvalue_called = &rvalue_callback_called,
                                   });
  TEST_ASSERT(overload_result.ok);
  TEST_ASSERT(!lvalue_callback_called);
  TEST_ASSERT(rvalue_callback_called);

  bool fold_lvalue_callback_called = false;
  bool fold_rvalue_callback_called = false;
  rund::kernel::u64 fold_cursor_sum = 0u;
  const rund::kernel::SkeletonResult fold_cursor_result =
      rund::kernel::fold(rund::kernel::space(2u, 2u), fold_cursor_sum,
                         FoldOverloadSensitiveIndexCallback{
                             .lvalue_called = &fold_lvalue_callback_called,
                             .rvalue_called = &fold_rvalue_callback_called,
                         });
  TEST_ASSERT(fold_cursor_result.ok);
  TEST_ASSERT(fold_cursor_sum == 22u);
  TEST_ASSERT(!fold_lvalue_callback_called);
  TEST_ASSERT(fold_rvalue_callback_called);

  bool zero_visited = false;
  const rund::kernel::SkeletonResult zero_result = rund::kernel::each(
      rund::kernel::space(0u, 8u), [&](auto) noexcept { zero_visited = true; });
  TEST_ASSERT(zero_result.ok);
  TEST_ASSERT(zero_result.visited_count == 0u);
  TEST_ASSERT(zero_result.plan_kind == rund::kernel::SkeletonPlanKind::Empty);
  TEST_ASSERT(!zero_visited);
  rund::kernel::i32 zero_acc = 7;
  const rund::kernel::SkeletonResult zero_fold =
      rund::kernel::fold(rund::kernel::space(0u), zero_acc,
                         [](rund::kernel::i32, auto) noexcept { return 0; });
  TEST_ASSERT(zero_fold.ok);
  TEST_ASSERT(zero_fold.visited_count == 0u);
  TEST_ASSERT(zero_fold.plan_kind == rund::kernel::SkeletonPlanKind::Empty);
  TEST_ASSERT(zero_acc == 7);
  return 0;
}

} // namespace program_skeleton_contract
