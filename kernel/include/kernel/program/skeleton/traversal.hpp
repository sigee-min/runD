#pragma once

#include <kernel/program/skeleton/callback.hpp>
#include <kernel/program/skeleton/model.hpp>

#include <utility>

namespace rund::kernel::skeleton_detail {

template <std::size_t Rank>
[[nodiscard]] constexpr Index<Rank> LinearIndexToRowMajor(
    const Space<Rank>& index_space,
    u64 linear) noexcept {
  if constexpr (Rank == 1u) {
    return Index<Rank>{linear};
  }
  Index<Rank> index{};
  for (std::size_t offset = 0u; offset < Rank; ++offset) {
    const std::size_t axis = Rank - 1u - offset;
    const u64 extent = index_space.extent[axis];
    index[axis] = linear % extent;
    linear /= extent;
  }
  return index;
}

template <std::size_t Rank>
constexpr void AdvanceRowMajorIndex(const Space<Rank>& index_space,
                                    Index<Rank>& index) noexcept {
  if constexpr (Rank == 1u) {
    ++index[0u];
  } else {
    for (std::size_t offset = 0u; offset < Rank; ++offset) {
      const std::size_t axis = Rank - 1u - offset;
      ++index[axis];
      if (index[axis] < index_space.extent[axis]) {
        return;
      }
      index[axis] = 0u;
    }
  }
}

template <std::size_t Rank>
[[nodiscard]] constexpr Index<Rank> CallbackIndexValue(
    const Index<Rank>& index) noexcept {
  return index;
}

template <std::size_t Rank, typename Callback>
  requires DirectCallback<Callback>
inline void ExecutePlan(const SkeletonPlan<Rank>& plan,
                        Callback&& callback) noexcept(noexcept(
                            std::declval<Callback&>()(
                                CallbackIndexValue(
                                    LinearIndexToRowMajor(plan.space, plan.begin))))) {
  if (plan.begin == plan.end) {
    return;
  }
  Index<Rank> index = LinearIndexToRowMajor(plan.space, plan.begin);
  for (u64 linear = plan.begin; linear < plan.end; ++linear) {
    callback(CallbackIndexValue(index));
    AdvanceRowMajorIndex(plan.space, index);
  }
}

template <std::size_t Rank, typename Accumulator, typename Callback>
  requires DirectCallback<Callback>
inline void ExecuteFold(const SkeletonPlan<Rank>& plan,
                        Accumulator& accumulator,
                        Callback&& callback) noexcept(noexcept(
                            accumulator = std::declval<Callback&>()(
                                accumulator,
                                CallbackIndexValue(
                                    LinearIndexToRowMajor(plan.space, plan.begin))))) {
  if (plan.begin == plan.end) {
    return;
  }
  Index<Rank> index = LinearIndexToRowMajor(plan.space, plan.begin);
  for (u64 linear = plan.begin; linear < plan.end; ++linear) {
    accumulator = callback(accumulator, CallbackIndexValue(index));
    AdvanceRowMajorIndex(plan.space, index);
  }
}

} // namespace rund::kernel::skeleton_detail
