#pragma once

#include <kernel/program/skeleton/plan/core.hpp>

namespace rund::kernel {

template <std::size_t Rank>
[[nodiscard]] constexpr SkeletonPlan<Rank> plan_fold(
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
                                   SkeletonPlanKind::SerialLeftFold);
}

} // namespace rund::kernel
