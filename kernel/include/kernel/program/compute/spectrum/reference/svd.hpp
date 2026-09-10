#pragma once

#include <kernel/program/compute/spectrum/reference/jacobi.hpp>

#include <algorithm>

namespace rund::kernel::spectrum_reference_detail {

template <typename S>
void BuildSymmetricAtA(const S *const src, S *const ata,
                       const SpectrumPlan &plan) noexcept {
  const u64 n = plan.cols;
  for (u64 i = 0u; i < n; ++i) {
    for (u64 j = i; j < n; ++j) {
      S sum = 0;
      for (u64 row = 0u; row < plan.rows; ++row) {
        sum = factor_reference_detail::AddSat(
            sum, factor_reference_detail::MulFixed(
                     src[Index(row, i, plan.rows, plan.cols, plan.layout)],
                     src[Index(row, j, plan.rows, plan.cols, plan.layout)],
                     plan.fixed_format));
      }
      ata[static_cast<std::size_t>(i * n + j)] = sum;
      if (i != j) {
        ata[static_cast<std::size_t>(j * n + i)] = sum;
      }
    }
  }
}

template <typename S>
[[nodiscard]] SpectrumStatus SvdBatch(
    const S *const input, S *const values, S *const vectors,
    const SpectrumPlan &plan, const u64 batch, S *const ata,
    S *const jacobi_vectors, S *const jacobi_values, u64 *const order,
    S *const u) {
  const u64 n = plan.cols;
  const S *const src = input + batch * plan.rows * plan.cols;
  BuildSymmetricAtA(src, ata, plan);
  JacobiResult eig = JacobiSymmetric(ata, jacobi_vectors, jacobi_values, n,
                                     plan.max_iterations, plan.fixed_format);
  if (!eig.converged) {
    return SpectrumStatus::NonConvergence;
  }
  for (u64 i = 0u; i < n; ++i) {
    const S value = jacobi_values[static_cast<std::size_t>(i)];
    jacobi_values[static_cast<std::size_t>(i)] =
        value < 0 ? S{0}
                  : factor_reference_detail::SqrtFixed(value,
                                                       plan.fixed_format);
    order[static_cast<std::size_t>(i)] = i;
  }
  for (u64 left = 0u; left < n; ++left) {
    for (u64 right = left + 1u; right < n; ++right) {
      if (jacobi_values[static_cast<std::size_t>(
              order[static_cast<std::size_t>(right)])] >
          jacobi_values[static_cast<std::size_t>(
              order[static_cast<std::size_t>(left)])]) {
        std::swap(order[static_cast<std::size_t>(left)],
                  order[static_cast<std::size_t>(right)]);
      }
    }
  }
  const u64 width = plan.value_count / plan.batch_count;
  S *const out = values + batch * width;
  for (u64 i = 0u; i < width; ++i) {
    const u64 value_index = i < n ? order[static_cast<std::size_t>(i)] : 0u;
    out[i] = i < n ? jacobi_values[static_cast<std::size_t>(value_index)]
                   : S{0};
  }
  if (vectors != nullptr && plan.vector_count != 0u) {
    const u64 vector_cols =
        plan.vectors == SpectrumVectors::Thin ? width : plan.rows;
    bool needs_basis = false;
    for (u64 col = 0u; col < vector_cols; ++col) {
      const bool has_singular = col < width;
      const u64 value_index =
          has_singular ? order[static_cast<std::size_t>(col)] : 0u;
      const S sigma = has_singular
                          ? jacobi_values[static_cast<std::size_t>(value_index)]
                          : S{0};
      if (!has_singular ||
          Magnitude(sigma) <= Magnitude(Epsilon<S>(plan.fixed_format))) {
        needs_basis = true;
        break;
      }
    }
    if (needs_basis) {
      std::fill_n(u, static_cast<std::size_t>(plan.rows * vector_cols), S{0});
    }
    for (u64 col = 0u; col < vector_cols; ++col) {
      const bool has_singular = col < width;
      const u64 value_index =
          has_singular ? order[static_cast<std::size_t>(col)] : 0u;
      const S sigma = has_singular
                          ? jacobi_values[static_cast<std::size_t>(value_index)]
                          : S{0};
      if (has_singular &&
          Magnitude(sigma) > Magnitude(Epsilon<S>(plan.fixed_format))) {
        for (u64 row = 0u; row < plan.rows; ++row) {
          S sum = 0;
          for (u64 k = 0u; k < plan.cols; ++k) {
            sum = factor_reference_detail::AddSat(
                sum, factor_reference_detail::MulFixed(
                         src[Index(row, k, plan.rows, plan.cols, plan.layout)],
                         jacobi_vectors[
                             static_cast<std::size_t>(k * n + value_index)],
                         plan.fixed_format));
          }
          u[static_cast<std::size_t>(row * vector_cols + col)] =
              factor_reference_detail::DivFixed(sum, sigma,
                                                plan.fixed_format);
        }
      } else {
        u[static_cast<std::size_t>(col * vector_cols + col)] =
            One<S>(plan.fixed_format);
      }
      OrthogonalizeColumn(u, plan.rows, vector_cols, col, plan.fixed_format);
      if (DotColumn(u, plan.rows, vector_cols, col, col, plan.fixed_format) ==
          0) {
        FillBasisColumn(u, plan.rows, vector_cols, col, plan.fixed_format);
      }
      NormalizeColumn(u, plan.rows, vector_cols, col, plan.fixed_format);
    }
    S *const vec = vectors + batch * plan.vector_count / plan.batch_count;
    for (u64 row = 0u; row < plan.rows; ++row) {
      for (u64 col = 0u; col < vector_cols; ++col) {
        vec[Index(row, col, plan.rows, vector_cols, plan.layout)] =
            u[static_cast<std::size_t>(row * vector_cols + col)];
      }
    }
  }
  return SpectrumStatus::Ok;
}

} // namespace rund::kernel::spectrum_reference_detail
