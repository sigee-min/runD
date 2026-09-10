#pragma once

#include <kernel/program/compute/solve/reference/base.hpp>

namespace rund::kernel {
namespace solve_reference_detail {

template <typename S>
[[nodiscard]] SolveStatus
SolveCholeskyBatch(const S *const factor, const S *const rhs, S *const output,
                   const SolvePlan &plan, const u64 batch) noexcept {
  const u64 n = plan.rows;
  const u64 rhs_cols = plan.rhs_cols;
  const S *const l = factor + batch * n * n;
  const S *const b = rhs + batch * n * rhs_cols;
  S *const x = output + batch * n * rhs_cols;
  for (u64 row = 0u; row < n; ++row) {
    for (u64 col = 0u; col < rhs_cols; ++col) {
      S sum = b[Index(row, col, n, rhs_cols, plan.layout)];
      for (u64 k = 0u; k < row; ++k) {
        sum = SubSat(sum, MulFixed(l[Index(row, k, n, n, plan.layout)],
                                   x[Index(k, col, n, rhs_cols, plan.layout)],
                                   plan.fixed_format));
      }
      const S diag = l[Index(row, row, n, n, plan.layout)];
      if (diag == 0) {
        return SolveStatus::NonSpd;
      }
      x[Index(row, col, n, rhs_cols, plan.layout)] =
          DivFixed(sum, diag, plan.fixed_format);
    }
  }
  for (u64 reverse = 0u; reverse < n; ++reverse) {
    const u64 row = n - 1u - reverse;
    for (u64 col = 0u; col < rhs_cols; ++col) {
      S sum = x[Index(row, col, n, rhs_cols, plan.layout)];
      for (u64 k = row + 1u; k < n; ++k) {
        sum = SubSat(sum, MulFixed(l[Index(k, row, n, n, plan.layout)],
                                   x[Index(k, col, n, rhs_cols, plan.layout)],
                                   plan.fixed_format));
      }
      const S diag = l[Index(row, row, n, n, plan.layout)];
      if (diag == 0) {
        return SolveStatus::NonSpd;
      }
      x[Index(row, col, n, rhs_cols, plan.layout)] =
          DivFixed(sum, diag, plan.fixed_format);
    }
  }
  return SolveStatus::Ok;
}

} // namespace solve_reference_detail
} // namespace rund::kernel
