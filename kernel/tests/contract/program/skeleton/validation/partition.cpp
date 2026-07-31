#include "local.hpp"

namespace program_skeleton_contract {
namespace {

int NarrowPartition() {
  const rund::kernel::Partition partition{
      .worker_index = 0u,
      .begin = 2u,
      .end = 6u,
  };
  rund::kernel::u64 partition_sum = 0u;
  const rund::kernel::SkeletonResult partition_result = rund::kernel::each(
      rund::kernel::space(8u), partition,
      [&](auto index) noexcept { partition_sum += index[0]; });
  TEST_ASSERT(partition_result.ok);
  TEST_ASSERT(partition_result.visited_count == 4u);
  TEST_ASSERT(partition_result.global_row_major_order);
  TEST_ASSERT(partition_result.partition_boundary_checked);
  TEST_ASSERT(partition_result.partition_boundary_aligned);
  TEST_ASSERT(partition_result.boundary_alignment_units == 1u);
  TEST_ASSERT(partition_result.physical_tile_units == 4u);
  TEST_ASSERT(partition_result.tile_order_preserves_row_major);
  TEST_ASSERT(partition_sum == 14u);
  const rund::kernel::SkeletonPlan<1u> partition_plan =
      rund::kernel::plan_each(rund::kernel::space(8u), partition);
  TEST_ASSERT(partition_plan.kind ==
              rund::kernel::SkeletonPlanKind::ContiguousLinear);
  TEST_ASSERT(partition_plan.partitioned);
  TEST_ASSERT(partition_plan.begin == 2u);
  TEST_ASSERT(partition_plan.end == 6u);
  TEST_ASSERT(partition_plan.boundary_alignment_checked);
  TEST_ASSERT(partition_plan.boundary_aligned);
  TEST_ASSERT(partition_plan.boundary_alignment_units == 1u);
  return 0;
}

int PartitionAlignment() {
  const rund::kernel::Partition aligned_partition{
      .worker_index = 0u,
      .begin = 4u,
      .end = 8u,
  };
  const rund::kernel::SkeletonResult aligned_partition_result =
      rund::kernel::each(rund::kernel::space(16u), aligned_partition,
                         rund::kernel::align(4u), [](auto) noexcept {});
  TEST_ASSERT(aligned_partition_result.ok);
  TEST_ASSERT(aligned_partition_result.partition_boundary_checked);
  TEST_ASSERT(aligned_partition_result.partition_boundary_aligned);
  TEST_ASSERT(aligned_partition_result.boundary_alignment_units == 4u);
  TEST_ASSERT(aligned_partition_result.physical_tile_units == 4u);
  TEST_ASSERT(aligned_partition_result.tile_order_preserves_row_major);

  const rund::kernel::Partition misaligned_partition{
      .worker_index = 0u,
      .begin = 2u,
      .end = 7u,
  };
  TEST_ASSERT(ExpectReason("skeleton_partition_boundary_misaligned",
                           rund::kernel::each(rund::kernel::space(16u),
                                              misaligned_partition,
                                              rund::kernel::align(4u),
                                              [](auto) noexcept {})) == 0);
  const rund::kernel::SkeletonPlan<1u> misaligned_plan =
      rund::kernel::plan_each(rund::kernel::space(16u), misaligned_partition,
                              rund::kernel::align(4u));
  TEST_ASSERT(misaligned_plan.kind == rund::kernel::SkeletonPlanKind::Invalid);
  TEST_ASSERT(std::string_view{misaligned_plan.reason} ==
              "skeleton_partition_boundary_misaligned");
  return 0;
}

int PartitionRejects() {
  const std::array<rund::kernel::u32, 1u> indirect{0u};
  const rund::kernel::Partition indirect_partition{
      .worker_index = 0u,
      .begin = 0u,
      .end = 1u,
      .packet_indices = indirect.data(),
      .packet_index_count = 1u,
  };
  TEST_ASSERT(ExpectReason("skeleton_indirect_partition",
                           rund::kernel::each(rund::kernel::space(8u),
                                              indirect_partition,
                                              [](auto) noexcept {})) == 0);

  const rund::kernel::Partition out_of_bounds{
      .worker_index = 0u,
      .begin = 7u,
      .end = 9u,
  };
  TEST_ASSERT(
      ExpectReason("skeleton_partition_out_of_bounds",
                   rund::kernel::each(rund::kernel::space(8u), out_of_bounds,
                                      [](auto) noexcept {})) == 0);

  TEST_ASSERT(ExpectReason("skeleton_negative_extent",
                           rund::kernel::each(rund::kernel::space(-1),
                                              [](auto) noexcept {})) == 0);

  const rund::kernel::u64 too_large =
      std::numeric_limits<rund::kernel::u64>::max();
  TEST_ASSERT(
      ExpectReason("skeleton_index_space_overflow",
                   rund::kernel::each(rund::kernel::space(too_large, 2u),
                                      [](auto) noexcept {})) == 0);
  return 0;
}

} // namespace

int SkeletonValidationPartition() {
  if (NarrowPartition() != 0) {
    return 1;
  }
  if (PartitionAlignment() != 0) {
    return 1;
  }
  return PartitionRejects();
}

} // namespace program_skeleton_contract
