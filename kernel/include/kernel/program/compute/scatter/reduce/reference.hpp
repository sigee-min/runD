#pragma once

#include <kernel/program/compute/fixed/arithmetic.hpp>
#include <kernel/program/compute/scatter/reduce/plan.hpp>

#include <algorithm>
#include <bit>
#include <limits>
#include <type_traits>

namespace rund::kernel {
namespace scatter_reduce_reference_detail {

[[nodiscard]] constexpr ScatterReduceResult
Reject(const ScatterReducePlan &plan, const u64 first_rejected_ordinal,
       const char *const reason) noexcept {
  return ScatterReduceResult{
      .element_count = plan.element_count,
      .output_count = plan.output_count,
      .first_rejected_ordinal = first_rejected_ordinal,
      .reason = reason,
  };
}

template <typename T>
[[nodiscard]] constexpr T AddWrap(const T lhs, const T rhs) noexcept {
  using U = std::make_unsigned_t<T>;
  const U value = static_cast<U>(lhs) + static_cast<U>(rhs);
  if constexpr (std::is_signed_v<T>) {
    return std::bit_cast<T>(value);
  } else {
    return value;
  }
}

template <typename T>
[[nodiscard]] constexpr T Identity(const ScatterReduceOp op) noexcept {
  return op == ScatterReduceOp::Sum
             ? T{0}
             : (op == ScatterReduceOp::Min ? std::numeric_limits<T>::max()
                                           : std::numeric_limits<T>::lowest());
}

template <typename T>
[[nodiscard]] inline ScatterReduceResult ReferenceScatterReduce(
    const T *const values, const u32 *const indices, T *const output,
    const u64 logical_count, const ScatterReducePlan &plan,
    const bool fixed, u32 *const sorted_indices,
    const u64 sorted_index_capacity) noexcept {
  if (!plan.ok) {
    return Reject(plan, 0u, plan.reason);
  }
  if (logical_count > plan.element_count) {
    return Reject(plan, plan.element_count,
                  "compute_scatter_reduce_count_out_of_range");
  }
  if ((logical_count != 0u && (values == nullptr || indices == nullptr)) ||
      output == nullptr ||
      (logical_count != 0u &&
       (sorted_indices == nullptr ||
        sorted_index_capacity < logical_count))) {
    return Reject(plan, 0u, "compute_scatter_reduce_buffer_invalid");
  }

  // Preflight is deliberately separate: neither identity initialization nor
  // a partial fold becomes visible if any authored target is invalid.
  for (u64 ordinal = 0u; ordinal < logical_count; ++ordinal) {
    if (static_cast<u64>(indices[ordinal]) >= plan.output_count) {
      return Reject(plan, ordinal,
                    "compute_scatter_reduce_index_out_of_range");
    }
    sorted_indices[ordinal] = indices[ordinal];
  }

  // The plan-owned sorted-index scratch is allocated during preparation.
  // Sorting only keys is sufficient for conflict evidence; the observable
  // fold below remains in source-ordinal order. Complexity is
  // O(N log N + N + O), warm allocations are zero, and no output is touched
  // before the complete index preflight succeeds.
  if (logical_count > 1u) {
    std::sort(sorted_indices, sorted_indices + logical_count);
  }
  u64 conflict_count = 0u;
  for (u64 ordinal = 1u; ordinal < logical_count; ++ordinal) {
    conflict_count +=
        sorted_indices[ordinal - 1u] == sorted_indices[ordinal] ? 1u : 0u;
  }

  for (u64 target = 0u; target < plan.output_count; ++target) {
    output[target] = Identity<T>(plan.op);
  }
  // Source ordinal is the tie-breaker. Iterating the source once therefore
  // defines the same fold order as stable (target, ordinal) sorting.
  for (u64 ordinal = 0u; ordinal < logical_count; ++ordinal) {
    T &selected = output[indices[ordinal]];
    const T value = values[ordinal];
    if (plan.op == ScatterReduceOp::Sum) {
      selected = fixed ? compute_fixed_detail::Narrow<T>(
                             static_cast<i128>(selected) + value,
                             plan.fixed_format.overflow)
                       : AddWrap(selected, value);
    } else if (plan.op == ScatterReduceOp::Min) {
      selected = value < selected ? value : selected;
    } else {
      selected = value > selected ? value : selected;
    }
  }
  return ScatterReduceResult{
      .element_count = logical_count,
      .output_count = plan.output_count,
      .first_rejected_ordinal = logical_count,
      .conflict_count = conflict_count,
      .ok = true,
      .reason = "ok",
  };
}

} // namespace scatter_reduce_reference_detail

[[nodiscard]] inline ScatterReduceResult ReferenceScatterReduceI32(
    const i32 *values, const u32 *indices, i32 *output,
    const u64 logical_count, const ScatterReducePlan &plan,
    u32 *sorted_indices, const u64 sorted_index_capacity) noexcept {
  return scatter_reduce_reference_detail::ReferenceScatterReduce(
      values, indices, output, logical_count, plan, false, sorted_indices,
      sorted_index_capacity);
}
[[nodiscard]] inline ScatterReduceResult ReferenceScatterReduceU32(
    const u32 *values, const u32 *indices, u32 *output,
    const u64 logical_count, const ScatterReducePlan &plan,
    u32 *sorted_indices, const u64 sorted_index_capacity) noexcept {
  return scatter_reduce_reference_detail::ReferenceScatterReduce(
      values, indices, output, logical_count, plan, false, sorted_indices,
      sorted_index_capacity);
}
[[nodiscard]] inline ScatterReduceResult ReferenceScatterReduceI64(
    const i64 *values, const u32 *indices, i64 *output,
    const u64 logical_count, const ScatterReducePlan &plan,
    u32 *sorted_indices, const u64 sorted_index_capacity) noexcept {
  return scatter_reduce_reference_detail::ReferenceScatterReduce(
      values, indices, output, logical_count, plan, false, sorted_indices,
      sorted_index_capacity);
}
[[nodiscard]] inline ScatterReduceResult ReferenceScatterReduceU64(
    const u64 *values, const u32 *indices, u64 *output,
    const u64 logical_count, const ScatterReducePlan &plan,
    u32 *sorted_indices, const u64 sorted_index_capacity) noexcept {
  return scatter_reduce_reference_detail::ReferenceScatterReduce(
      values, indices, output, logical_count, plan, false, sorted_indices,
      sorted_index_capacity);
}
[[nodiscard]] inline ScatterReduceResult ReferenceScatterReduceFixedI32(
    const i32 *values, const u32 *indices, i32 *output,
    const u64 logical_count, const ScatterReducePlan &plan,
    u32 *sorted_indices, const u64 sorted_index_capacity) noexcept {
  return scatter_reduce_reference_detail::ReferenceScatterReduce(
      values, indices, output, logical_count, plan, true, sorted_indices,
      sorted_index_capacity);
}
[[nodiscard]] inline ScatterReduceResult ReferenceScatterReduceFixedI64(
    const i64 *values, const u32 *indices, i64 *output,
    const u64 logical_count, const ScatterReducePlan &plan,
    u32 *sorted_indices, const u64 sorted_index_capacity) noexcept {
  return scatter_reduce_reference_detail::ReferenceScatterReduce(
      values, indices, output, logical_count, plan, true, sorted_indices,
      sorted_index_capacity);
}

} // namespace rund::kernel
