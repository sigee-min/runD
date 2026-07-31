#pragma once

#include <kernel/program/skeleton/plan/core.hpp>

namespace rund::kernel {

template <std::size_t Rank>
[[nodiscard]] constexpr SkeletonPlan<Rank> plan_each(
    const Space<Rank>& index_space) noexcept {
  static_assert(Rank > 0u);
  if (const char* const reason = skeleton_detail::ValidateSpaceReason(index_space);
      reason != nullptr) {
    return skeleton_detail::InvalidPlan(index_space, reason);
  }
  const u64 units = skeleton_detail::UnitCount(index_space);
  return skeleton_detail::MakePlan(index_space,
                                   0u,
                                   units,
                                   false,
                                   SkeletonPlanKind::ContiguousLinear);
}

template <std::size_t Rank>
[[nodiscard]] constexpr SkeletonPlan<Rank> plan_each(
    const Space<Rank>& index_space,
    const Partition& partition,
    const Alignment boundary_alignment) noexcept {
  static_assert(Rank > 0u);
  if (const char* const reason =
          skeleton_detail::ValidatePartitionSpaceReason(index_space);
      reason != nullptr) {
    return skeleton_detail::InvalidPlan(index_space, reason);
  }
  if (partition.packet_indices != nullptr) {
    return skeleton_detail::InvalidPlan(index_space, "skeleton_indirect_partition");
  }
  const u64 units = skeleton_detail::UnitCount(index_space);
  if (partition.begin > units || partition.end > units || partition.end < partition.begin) {
    return skeleton_detail::InvalidPlan(index_space, "skeleton_partition_out_of_bounds");
  }
  if (const char* const reason =
          skeleton_detail::ValidatePartitionBoundaryAlignmentReason(partition,
                                                                    boundary_alignment);
      reason != nullptr) {
    return skeleton_detail::InvalidPlan(index_space, reason);
  }
  return skeleton_detail::MakePlan(index_space,
                                   partition.begin,
                                   partition.end,
                                   true,
                                   SkeletonPlanKind::ContiguousLinear,
                                   boundary_alignment);
}

template <std::size_t Rank>
[[nodiscard]] constexpr SkeletonPlan<Rank> plan_each(
    const Space<Rank>& index_space,
    const Partition& partition) noexcept {
  return plan_each(index_space, partition, Alignment{});
}

} // namespace rund::kernel
