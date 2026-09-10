#pragma once

#include <kernel/program/compute/spectrum/reference/jacobi.hpp>

namespace rund::kernel::spectrum_reference_detail {

template <typename S>
[[nodiscard]] SpectrumStatus EigenBatch(const S *const input, S *const values,
                                        S *const vectors,
                                        const SpectrumPlan &plan,
                                        const u64 batch, S *const a,
                                        S *const jacobi_vectors,
                                        S *const jacobi_values) {
  const S *const src = input + batch * plan.rows * plan.cols;
  for (u64 row = 0u; row < plan.rows; ++row) {
    for (u64 col = 0u; col < plan.cols; ++col) {
      a[static_cast<std::size_t>(row * plan.rows + col)] =
          src[Index(row, col, plan.rows, plan.cols, plan.layout)];
    }
  }
  JacobiResult eig = JacobiSymmetric(a, jacobi_vectors, jacobi_values,
                                     plan.rows, plan.max_iterations,
                                     plan.fixed_format);
  if (!eig.converged) {
    return SpectrumStatus::NonConvergence;
  }
  S *const out = values + batch * plan.rows;
  for (u64 i = 0u; i < plan.rows; ++i) {
    out[i] = jacobi_values[static_cast<std::size_t>(i)];
  }
  if (vectors != nullptr && plan.vector_count != 0u) {
    S *const vec = vectors + batch * plan.vector_count / plan.batch_count;
    const u64 cols = plan.rows;
    for (u64 row = 0u; row < plan.rows; ++row) {
      for (u64 col = 0u; col < cols; ++col) {
        vec[Index(row, col, plan.rows, cols, plan.layout)] =
            jacobi_vectors[static_cast<std::size_t>(row * plan.rows + col)];
      }
    }
  }
  return SpectrumStatus::Ok;
}

} // namespace rund::kernel::spectrum_reference_detail
