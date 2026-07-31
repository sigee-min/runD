#include "local.hpp"

namespace program_skeleton_contract {

int SkeletonShapeElementwise() {
  std::array<rund::kernel::i32, 8u> lhs{};
  std::array<rund::kernel::i32, 8u> rhs{};
  std::array<rund::kernel::i32, 8u> out{};
  for (std::size_t index = 0u; index < lhs.size(); ++index) {
    lhs[index] = static_cast<rund::kernel::i32>(index);
    rhs[index] = static_cast<rund::kernel::i32>(index * 3u);
  }

  auto lhs1 = rund::kernel::view<rund::kernel::i32>(
      lhs.data(), rund::kernel::Index<1u>{lhs.size()});
  auto rhs1 = rund::kernel::view<rund::kernel::i32>(
      rhs.data(), rund::kernel::Index<1u>{rhs.size()});
  auto out1 = rund::kernel::view<rund::kernel::i32>(
      out.data(), rund::kernel::Index<1u>{out.size()});
  TEST_ASSERT(lhs1);
  TEST_ASSERT(rhs1);
  TEST_ASSERT(out1);

  rund::kernel::u64 reduction = 0u;
  const rund::kernel::SkeletonResult elementwise = rund::kernel::each(
      rund::kernel::space(lhs.size()), [&](auto index) noexcept {
        const rund::kernel::i32 value =
            rund::math32::detail::ScalarAddWrap(lhs1(index), rhs1(index));
        out1(index) = value;
        reduction += static_cast<rund::kernel::u32>(value);
      });
  TEST_ASSERT(elementwise.ok);
  TEST_ASSERT(elementwise.visited_count == lhs.size());
  TEST_ASSERT(elementwise.plan_kind ==
              rund::kernel::SkeletonPlanKind::ContiguousLinear);
  TEST_ASSERT(elementwise.deterministic_traversal);
  TEST_ASSERT(elementwise.contiguous_linear_traversal);
  TEST_ASSERT(elementwise.global_row_major_order);
  TEST_ASSERT(!elementwise.partition_boundary_checked);
  TEST_ASSERT(!elementwise.partition_boundary_aligned);
  TEST_ASSERT(elementwise.boundary_alignment_units == 1u);
  TEST_ASSERT(elementwise.physical_tile_units == lhs.size());
  TEST_ASSERT(elementwise.tile_order_preserves_row_major);
  TEST_ASSERT(elementwise.vectorizable_shape);
  TEST_ASSERT(!elementwise.simd_measured);
  TEST_ASSERT(std::string_view{elementwise.vector_reason} ==
              "validated_contiguous_plan");
  const rund::kernel::SkeletonPlan<1u> elementwise_plan =
      rund::kernel::plan_each(rund::kernel::space(lhs.size()));
  TEST_ASSERT(elementwise_plan.kind ==
              rund::kernel::SkeletonPlanKind::ContiguousLinear);
  TEST_ASSERT(elementwise_plan.unit_count == lhs.size());
  TEST_ASSERT(elementwise_plan.physical_tile_units == lhs.size());
  TEST_ASSERT(elementwise_plan.tile_order_preserves_row_major);
  TEST_ASSERT(elementwise_plan.vectorizable_shape);
  TEST_ASSERT(!elementwise_plan.simd_measured);
  for (std::size_t index = 0u; index < out.size(); ++index) {
    TEST_ASSERT(out[index] ==
                rund::math32::detail::ScalarAddWrap(lhs[index], rhs[index]));
  }
  return 0;
}

} // namespace program_skeleton_contract
