#pragma once

#include <kernel/program/skeleton/validation.hpp>

namespace rund::kernel::skeleton_detail {

[[nodiscard]] constexpr SkeletonResult InvalidResult(const char* const reason) noexcept {
  return SkeletonResult{.reason = reason};
}

template <std::size_t Rank>
[[nodiscard]] constexpr SkeletonPlan<Rank> InvalidPlan(
    const Space<Rank>& index_space,
    const char* const reason) noexcept {
  return SkeletonPlan<Rank>{
      .space = index_space,
      .reason = reason,
  };
}

template <std::size_t Rank>
[[nodiscard]] constexpr SkeletonPlan<Rank> MakePlan(
    const Space<Rank>& index_space,
    const u64 begin,
    const u64 end,
    const bool partitioned,
    const SkeletonPlanKind non_empty_kind,
    const Alignment boundary_alignment = Alignment{}) noexcept {
  const u64 unit_count = end - begin;
  return SkeletonPlan<Rank>{
      .kind = begin == end ? SkeletonPlanKind::Empty : non_empty_kind,
      .space = index_space,
      .begin = begin,
      .end = end,
      .unit_count = unit_count,
      .partitioned = partitioned,
      .global_row_major_order = true,
      .boundary_alignment_checked = partitioned,
      .boundary_aligned = partitioned ? true : false,
      .boundary_alignment_units = boundary_alignment.units,
      .physical_tile_units = unit_count,
      .tile_order_preserves_row_major = true,
      .vectorizable_shape = non_empty_kind == SkeletonPlanKind::ContiguousLinear,
      .simd_measured = false,
      .reason = "pass",
  };
}

template <std::size_t Rank>
[[nodiscard]] constexpr SkeletonResult PlanResult(const SkeletonPlan<Rank>& plan) noexcept {
  if (plan.kind == SkeletonPlanKind::Invalid) {
    return InvalidResult(plan.reason);
  }
  return SkeletonResult{
      .ok = true,
      .reason = "pass",
      .visited_count = plan.unit_count,
      .plan_kind = plan.kind,
      .deterministic_traversal = true,
      .contiguous_linear_traversal =
          plan.kind == SkeletonPlanKind::Empty ||
          plan.kind == SkeletonPlanKind::ContiguousLinear ||
          plan.kind == SkeletonPlanKind::SerialLeftFold,
      .global_row_major_order = plan.global_row_major_order,
      .partition_boundary_checked = plan.boundary_alignment_checked,
      .partition_boundary_aligned = plan.boundary_aligned,
      .boundary_alignment_units = plan.boundary_alignment_units,
      .physical_tile_units = plan.physical_tile_units,
      .tile_order_preserves_row_major = plan.tile_order_preserves_row_major,
      .vectorizable_shape = plan.vectorizable_shape,
      .simd_measured = plan.simd_measured,
      .vector_reason = plan.vectorizable_shape ? "validated_contiguous_plan"
                                               : "not_vectorizable_plan",
  };
}

} // namespace rund::kernel::skeleton_detail
