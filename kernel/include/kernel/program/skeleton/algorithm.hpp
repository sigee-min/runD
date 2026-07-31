#pragma once

#include <kernel/program/skeleton/callback.hpp>
#include <kernel/program/skeleton/plan.hpp>
#include <kernel/program/skeleton/traversal.hpp>

#include <utility>

namespace rund::kernel {

template <std::size_t Rank, typename Callback>
  requires skeleton_detail::DirectCallback<Callback>
[[nodiscard]] inline SkeletonResult each(
    const Space<Rank>& index_space,
    Callback&& callback) {
  const SkeletonPlan<Rank> plan = plan_each(index_space);
  if (plan.kind == SkeletonPlanKind::Invalid) {
    return skeleton_detail::PlanResult(plan);
  }
  skeleton_detail::ExecutePlan(plan, std::forward<Callback>(callback));
  return skeleton_detail::PlanResult(plan);
}

template <std::size_t Rank, typename Callback>
  requires skeleton_detail::DirectCallback<Callback>
[[nodiscard]] inline SkeletonResult each(
    const Space<Rank>& index_space,
    const Partition& partition,
    const Alignment boundary_alignment,
    Callback&& callback) {
  const SkeletonPlan<Rank> plan = plan_each(index_space,
                                            partition,
                                            boundary_alignment);
  if (plan.kind == SkeletonPlanKind::Invalid) {
    return skeleton_detail::PlanResult(plan);
  }
  skeleton_detail::ExecutePlan(plan, std::forward<Callback>(callback));
  return skeleton_detail::PlanResult(plan);
}

template <std::size_t Rank, typename Callback>
  requires skeleton_detail::DirectCallback<Callback>
[[nodiscard]] inline SkeletonResult each(
    const Space<Rank>& index_space,
    const Partition& partition,
    Callback&& callback) {
  return each(index_space,
              partition,
              Alignment{},
              std::forward<Callback>(callback));
}

template <std::size_t Rank, typename Accumulator, typename Callback>
  requires skeleton_detail::DirectCallback<Callback>
[[nodiscard]] inline SkeletonResult fold(
    const Space<Rank>& index_space,
    Accumulator& accumulator,
    Callback&& callback) noexcept(noexcept(
        accumulator = std::declval<Callback&>()(
            accumulator,
            skeleton_detail::CallbackIndexValue(
                skeleton_detail::LinearIndexToRowMajor(index_space, 0u))))) {
  static_assert(Rank > 0u);
  const SkeletonPlan<Rank> plan = plan_fold(index_space);
  if (plan.kind == SkeletonPlanKind::Invalid) {
    return skeleton_detail::PlanResult(plan);
  }
  skeleton_detail::ExecuteFold(plan, accumulator, std::forward<Callback>(callback));
  return skeleton_detail::PlanResult(plan);
}

} // namespace rund::kernel
