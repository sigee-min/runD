#pragma once

#include <kernel/program/compute/solve/reference/base.hpp>

namespace rund::kernel {
namespace solve_reference_detail {

template <typename S>
[[nodiscard]] SolveStatus
SolveLuBatch(const S *const factor, const u32 *const aux, const S *const rhs,
             S *const output, const SolvePlan &plan, const u64 batch) noexcept {
  const u64 n = plan.rows;
  const u64 rhs_cols = plan.rhs_cols;
  const S *const lu = factor + batch * n * n;
  const u32 *const pivots = aux == nullptr ? nullptr : aux + batch * n;
  const S *const b = rhs + batch * n * rhs_cols;
  S *const x = output + batch * n * rhs_cols;
  for (u64 row = 0u; row < n; ++row) {
    for (u64 col = 0u; col < rhs_cols; ++col) {
      x[Index(row, col, n, rhs_cols, plan.layout)] =
          b[Index(row, col, n, rhs_cols, plan.layout)];
    }
  }
  if (pivots != nullptr) {
    for (u64 k = 0u; k < n; ++k) {
      const u64 pivot = pivots[k];
      if (pivot >= n) {
        return SolveStatus::Singular;
      }
      if (pivot == k) {
        continue;
      }
      for (u64 col = 0u; col < rhs_cols; ++col) {
        const u64 lhs = Index(k, col, n, rhs_cols, plan.layout);
        const u64 rhs_index = Index(pivot, col, n, rhs_cols, plan.layout);
        const S tmp = x[lhs];
        x[lhs] = x[rhs_index];
        x[rhs_index] = tmp;
      }
    }
  }
  for (u64 row = 0u; row < n; ++row) {
    for (u64 col = 0u; col < rhs_cols; ++col) {
      S sum = x[Index(row, col, n, rhs_cols, plan.layout)];
      for (u64 k = 0u; k < row; ++k) {
        sum = SubSat(sum, MulFixed(lu[Index(row, k, n, n, plan.layout)],
                                   x[Index(k, col, n, rhs_cols, plan.layout)],
                                   plan.fixed_format));
      }
      x[Index(row, col, n, rhs_cols, plan.layout)] = sum;
    }
  }
  for (u64 reverse = 0u; reverse < n; ++reverse) {
    const u64 row = n - 1u - reverse;
    const S diag = lu[Index(row, row, n, n, plan.layout)];
    if (diag == 0) {
      return SolveStatus::Singular;
    }
    for (u64 col = 0u; col < rhs_cols; ++col) {
      S sum = x[Index(row, col, n, rhs_cols, plan.layout)];
      for (u64 k = row + 1u; k < n; ++k) {
        sum = SubSat(sum, MulFixed(lu[Index(row, k, n, n, plan.layout)],
                                   x[Index(k, col, n, rhs_cols, plan.layout)],
                                   plan.fixed_format));
      }
      x[Index(row, col, n, rhs_cols, plan.layout)] =
          DivFixed(sum, diag, plan.fixed_format);
    }
  }
  return SolveStatus::Ok;
}

} // namespace solve_reference_detail
} // namespace rund::kernel
