#pragma once

#include <kernel/program/compute/scatter/reduce/cpu/plan.hpp>
#include <kernel/program/compute/scatter/reduce/reference.hpp>

namespace rund::kernel {

template <typename T>
[[nodiscard]] inline ScatterReduceResult
ExecuteScatterReduceCpu(const T *const values, const u32 *const indices,
                        T *const output, const u64 logical_count,
                        const ScatterReducePlan &plan,
                        const ScatterReduceCpuPlan &cpu, u32 *const scratch,
                        const u64 scratch_capacity) noexcept {
  using namespace scatter_reduce_reference_detail;
  if (!plan.ok) {
    return Reject(plan, 0u, plan.reason);
  }
  if (logical_count > plan.element_count) {
    return Reject(plan, plan.element_count,
                  "compute_scatter_reduce_count_out_of_range");
  }
  if (cpu != PlanScatterReduceCpu(plan) || output == nullptr ||
      (logical_count != 0u &&
       (values == nullptr || indices == nullptr || scratch == nullptr ||
        scratch_capacity < cpu.scratch_words))) {
    return Reject(plan, 0u, "compute_scatter_reduce_buffer_invalid");
  }
  const bool fixed = plan.domain == ComputeDomain::Fixed;
  if (cpu.strategy == ScatterReduceCpuStrategy::SortedKeys) {
    return ReferenceScatterReduce(values, indices, output, logical_count, plan,
                                  fixed, scratch, scratch_capacity);
  }

  u64 occupied = 0u;
  if (logical_count != 0u) {
    std::fill_n(scratch, cpu.scratch_words, u32{0u});
    u32 active_word = ~u32{0u};
    u32 original = 0u;
    u32 pending = 0u;
    const auto flush = [&]() noexcept {
      if (pending != original) {
        scratch[active_word] = pending;
        occupied += static_cast<u64>(std::popcount(pending ^ original));
      }
    };
    for (u64 ordinal = 0u; ordinal < logical_count; ++ordinal) {
      const u32 target = indices[ordinal];
      if (static_cast<u64>(target) >= plan.output_count) {
        return Reject(plan, ordinal,
                      "compute_scatter_reduce_index_out_of_range");
      }
      const u32 word = target / 32u;
      if (word != active_word) {
        flush();
        active_word = word;
        original = scratch[word];
        pending = original;
      }
      pending |= u32{1u} << (target % 32u);
    }
    flush();
  }
  // No caller output was touched before the complete preflight. In
  // particular, invalid scratch may be dirty but cannot affect the next run.
  return FoldScatterReduce(values, indices, output, logical_count, plan, fixed,
                           logical_count - occupied);
}

} // namespace rund::kernel
