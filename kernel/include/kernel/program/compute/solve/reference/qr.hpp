#pragma once

#include <kernel/program/compute/solve/reference/base.hpp>

#include <cstddef>

namespace rund::kernel {
namespace solve_reference_detail {

template <typename S>
[[nodiscard]] SolveStatus
SolveQrWorkspaceBatch(const S *const q, const S *const r,
                      const MatrixLayout factor_layout, const S *const rhs,
                      S *const output, const SolvePlan &plan, const u64 batch,
                      S *const y) {
  const u64 n = plan.rows;
  const u64 rhs_cols = plan.rhs_cols;
  const S *const b = rhs + batch * n * rhs_cols;
  S *const x = output + batch * n * rhs_cols;
  if (q == nullptr || r == nullptr || y == nullptr) {
    return SolveStatus::InvalidScaling;
  }
  for (u64 row = 0u; row < n; ++row) {
    for (u64 col = 0u; col < rhs_cols; ++col) {
      S sum = 0;
      for (u64 k = 0u; k < n; ++k) {
        sum = factor_reference_detail::AddSat(
            sum, MulFixed(q[Index(k, row, n, n, factor_layout)],
                          b[Index(k, col, n, rhs_cols, plan.layout)],
                          plan.fixed_format));
      }
      y[static_cast<std::size_t>(Index(row, col, n, rhs_cols, plan.layout))] =
          sum;
    }
  }
  for (u64 reverse = 0u; reverse < n; ++reverse) {
    const u64 row = n - 1u - reverse;
    const S diag = r[Index(row, row, n, n, factor_layout)];
    if (diag == 0) {
      return SolveStatus::Singular;
    }
    for (u64 col = 0u; col < rhs_cols; ++col) {
      S sum = y[static_cast<std::size_t>(
          Index(row, col, n, rhs_cols, plan.layout))];
      for (u64 k = row + 1u; k < n; ++k) {
        sum = SubSat(sum, MulFixed(r[Index(row, k, n, n, factor_layout)],
                                   x[Index(k, col, n, rhs_cols, plan.layout)],
                                   plan.fixed_format));
      }
      x[Index(row, col, n, rhs_cols, plan.layout)] =
          DivFixed(sum, diag, plan.fixed_format);
    }
  }
  return SolveStatus::Ok;
}

template <typename S>
[[nodiscard]] SolveStatus
SolveQrBatch(const S *const factor, const S *const rhs, S *const output,
             const SolvePlan &plan, const u64 batch, S *const y) {
  const u64 factor_stride = plan.factor_count / plan.batch_count;
  const S *const q = factor + batch * factor_stride;
  return SolveQrWorkspaceBatch(q, q + plan.rows * plan.rows, plan.layout, rhs,
                               output, plan, batch, y);
}

} // namespace solve_reference_detail
} // namespace rund::kernel
