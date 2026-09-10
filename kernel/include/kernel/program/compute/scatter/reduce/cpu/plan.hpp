#pragma once

#include <kernel/program/compute/scatter/reduce/plan.hpp>

namespace rund::kernel {

enum class ScatterReduceCpuStrategy : u8 { SortedKeys, SeenBitmap };

struct ScatterReduceCpuPlan final {
  ScatterReduceCpuStrategy strategy{ScatterReduceCpuStrategy::SortedKeys};
  u64 scratch_words{};

  [[nodiscard]] constexpr bool
  operator==(const ScatterReduceCpuPlan &) const noexcept = default;
};

// The dense strategy never reserves more scratch than the bounded sorted
// policy. This choice depends only on admitted capacity, never on warm data.
[[nodiscard]] constexpr ScatterReduceCpuPlan
PlanScatterReduceCpu(const ScatterReducePlan &plan) noexcept {
  if (!plan.ok) {
    return {};
  }
  const u64 bitmap_words =
      plan.output_count / 32u + (plan.output_count % 32u != 0u ? 1u : 0u);
  return bitmap_words <= plan.element_count
             ? ScatterReduceCpuPlan{ScatterReduceCpuStrategy::SeenBitmap,
                                    bitmap_words}
             : ScatterReduceCpuPlan{ScatterReduceCpuStrategy::SortedKeys,
                                    plan.element_count};
}

} // namespace rund::kernel
