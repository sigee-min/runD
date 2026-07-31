#pragma once

#include <kernel/program/compute/factor/reference.hpp>
#include <kernel/program/compute/spectrum/plan.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

namespace rund::kernel {
namespace spectrum_reference_detail {

[[nodiscard]] constexpr u64 Index(const u64 row, const u64 col, const u64 rows,
                                  const u64 cols,
                                  const MatrixLayout layout) noexcept {
  return factor_reference_detail::Index(row, col, rows, cols, layout);
}

template <typename S>
void MarkStatus(u32* const status, const u64 batch,
                const SpectrumStatus value) noexcept {
  if (status != nullptr) { status[batch] = static_cast<u32>(value); }
}

[[nodiscard]] inline SpectrumResult RecordFailure(
    SpectrumResult result,
    const u64 batch,
    const SpectrumStatus status) noexcept {
  if (result.failed_batches == 0u) {
    result.first_failed_batch = batch;
    result.first_status = status;
  }
  ++result.failed_batches;
  return result;
}

struct JacobiResult {
  bool converged = false;
};

template <typename S>
[[nodiscard]] constexpr S One(const ComputeFixedFormat format) noexcept {
  return format.fraction_bits == sizeof(S) * 8u - 1u
             ? std::numeric_limits<S>::max()
             : static_cast<S>(static_cast<std::make_unsigned_t<S>>(1u)
                              << format.fraction_bits);
}

template <typename S>
[[nodiscard]] constexpr S Epsilon(const ComputeFixedFormat format) noexcept {
  const S one = One<S>(format);
  return one > static_cast<S>(1u << 20u) ? static_cast<S>(one >> 20u) : S{1};
}

template <typename S>
[[nodiscard]] constexpr S Quarter(const ComputeFixedFormat format) noexcept {
  return static_cast<S>(One<S>(format) >> 2u);
}

template <typename S>
[[nodiscard]] constexpr S Sixteenth(const ComputeFixedFormat format) noexcept {
  return static_cast<S>(One<S>(format) >> 4u);
}

template <typename S>
[[nodiscard]] constexpr u128 Magnitude(const S value) noexcept {
  return value < 0 ? static_cast<u128>(-static_cast<i128>(value))
                   : static_cast<u128>(value);
}

template <typename S>
[[nodiscard]] constexpr S NormalizeDivide(const S value,
                                          const S norm,
                                          const ComputeFixedFormat format) noexcept {
  if (Magnitude(value) == Magnitude(norm)) {
    return (value < 0) != (norm < 0) ? -One<S>(format)
                                     : One<S>(format);
  }
  return factor_reference_detail::DivFixed(value, norm, format);
}

template <typename S>
[[nodiscard]] constexpr S AbsSat(const S value) noexcept {
  const u128 magnitude = Magnitude(value);
  return magnitude > static_cast<u128>(std::numeric_limits<S>::max())
             ? std::numeric_limits<S>::max()
             : static_cast<S>(magnitude);
}

template <typename S>
[[nodiscard]] inline S DotColumn(const S* const matrix,
                                 const u64 rows,
                                 const u64 cols,
                                 const u64 lhs,
                                 const u64 rhs,
                                 const ComputeFixedFormat format) noexcept {
  S dot = 0;
  for (u64 row = 0u; row < rows; ++row) {
    dot = factor_reference_detail::AddSat(
        dot, factor_reference_detail::MulFixed(
                 matrix[static_cast<std::size_t>(row * cols + lhs)],
                 matrix[static_cast<std::size_t>(row * cols + rhs)],
                 format));
  }
  return dot;
}

template <typename S>
inline void NormalizeColumn(S* const matrix, const u64 rows, const u64 cols,
                            const u64 col,
                            const ComputeFixedFormat format) noexcept {
  const S squared = DotColumn(matrix, rows, cols, col, col, format);
  const S norm = factor_reference_detail::SqrtFixed(squared, format);
  if (norm == 0) { return; }
  for (u64 row = 0u; row < rows; ++row) {
    S& value = matrix[static_cast<std::size_t>(row * cols + col)];
    value = NormalizeDivide(value, norm, format);
  }
}

template <typename S>
inline void OrthogonalizeColumn(S* const matrix, const u64 rows,
                                const u64 cols, const u64 col,
                                const ComputeFixedFormat format) noexcept {
  for (u64 prev = 0u; prev < col; ++prev) {
    const S dot = DotColumn(matrix, rows, cols, prev, col, format);
    for (u64 row = 0u; row < rows; ++row) {
      S& value = matrix[static_cast<std::size_t>(row * cols + col)];
      value = factor_reference_detail::SubSat(
          value, factor_reference_detail::MulFixed(
                     dot, matrix[static_cast<std::size_t>(row * cols + prev)],
                     format));
    }
  }
}

template <typename S>
inline void FillBasisColumn(S* const matrix, const u64 rows, const u64 cols,
                            const u64 col,
                            const ComputeFixedFormat format) noexcept {
  for (u64 row = 0u; row < rows; ++row) {
    matrix[static_cast<std::size_t>(row * cols + col)] =
        row == (col % rows) ? One<S>(format) : S{0};
  }
}

template <typename S>
[[nodiscard]] inline JacobiResult JacobiSymmetric(
    S* const a, S* const v, S* const values, const u64 n,
    const u32 max_iterations, const ComputeFixedFormat format) {
  if (a == nullptr || v == nullptr || values == nullptr) {
    return {};
  }
  std::fill_n(v, static_cast<std::size_t>(n * n), S{0});
  for (u64 i = 0u; i < n; ++i) {
    v[static_cast<std::size_t>(i * n + i)] = One<S>(format);
  }
  bool converged = false;
  for (u32 iteration = 0u; iteration < max_iterations; ++iteration) {
    u64 p = 0u;
    u64 q = 1u;
    S best = 0;
    for (u64 row = 0u; row < n; ++row) {
      for (u64 col = row + 1u; col < n; ++col) {
        const S candidate = a[static_cast<std::size_t>(row * n + col)];
        if (Magnitude(candidate) > Magnitude(best)) {
          best = candidate;
          p = row;
          q = col;
        }
      }
    }
    if (Magnitude(best) <= Magnitude(Epsilon<S>(format))) {
      converged = true;
      break;
    }
    const S app = a[static_cast<std::size_t>(p * n + p)];
    const S aqq = a[static_cast<std::size_t>(q * n + q)];
    const S apq = a[static_cast<std::size_t>(p * n + q)];
    const S tau = factor_reference_detail::DivFixed(
        factor_reference_detail::SubSat(aqq, app),
        factor_reference_detail::AddSat(apq, apq), format);
    const S tau_scaled = factor_reference_detail::MulFixed(
        tau, Quarter<S>(format), format);
    const S root_scaled = factor_reference_detail::SqrtFixed(
        factor_reference_detail::AddSat(
            Sixteenth<S>(format),
            factor_reference_detail::MulFixed(tau_scaled, tau_scaled,
                                              format)),
        format);
    const S sign_scaled =
        tau < 0 ? static_cast<S>(-Quarter<S>(format))
                : Quarter<S>(format);
    const S t = factor_reference_detail::DivFixed(
        sign_scaled,
        factor_reference_detail::AddSat(
            factor_reference_detail::MulFixed(
                AbsSat(tau), Quarter<S>(format), format),
            root_scaled),
        format);
    const S t_scaled = factor_reference_detail::MulFixed(
        t, Quarter<S>(format), format);
    const S c = factor_reference_detail::DivFixed(
        Quarter<S>(format),
        factor_reference_detail::SqrtFixed(
            factor_reference_detail::AddSat(
                Sixteenth<S>(format),
                factor_reference_detail::MulFixed(t_scaled, t_scaled,
                                                  format)),
            format),
        format);
    const S s = factor_reference_detail::MulFixed(t, c, format);
    for (u64 k = 0u; k < n; ++k) {
      const S akp = a[static_cast<std::size_t>(k * n + p)];
      const S akq = a[static_cast<std::size_t>(k * n + q)];
      a[static_cast<std::size_t>(k * n + p)] = factor_reference_detail::SubSat(
          factor_reference_detail::MulFixed(c, akp, format),
          factor_reference_detail::MulFixed(s, akq, format));
      a[static_cast<std::size_t>(k * n + q)] = factor_reference_detail::AddSat(
          factor_reference_detail::MulFixed(s, akp, format),
          factor_reference_detail::MulFixed(c, akq, format));
    }
    for (u64 k = 0u; k < n; ++k) {
      const S apk = a[static_cast<std::size_t>(p * n + k)];
      const S aqk = a[static_cast<std::size_t>(q * n + k)];
      a[static_cast<std::size_t>(p * n + k)] = factor_reference_detail::SubSat(
          factor_reference_detail::MulFixed(c, apk, format),
          factor_reference_detail::MulFixed(s, aqk, format));
      a[static_cast<std::size_t>(q * n + k)] = factor_reference_detail::AddSat(
          factor_reference_detail::MulFixed(s, apk, format),
          factor_reference_detail::MulFixed(c, aqk, format));
    }
    for (u64 k = 0u; k < n; ++k) {
      const S vkp = v[static_cast<std::size_t>(k * n + p)];
      const S vkq = v[static_cast<std::size_t>(k * n + q)];
      v[static_cast<std::size_t>(k * n + p)] = factor_reference_detail::SubSat(
          factor_reference_detail::MulFixed(c, vkp, format),
          factor_reference_detail::MulFixed(s, vkq, format));
      v[static_cast<std::size_t>(k * n + q)] = factor_reference_detail::AddSat(
          factor_reference_detail::MulFixed(s, vkp, format),
          factor_reference_detail::MulFixed(c, vkq, format));
    }
  }
  for (u64 i = 0u; i < n; ++i) {
    values[static_cast<std::size_t>(i)] =
        a[static_cast<std::size_t>(i * n + i)];
  }
  return JacobiResult{.converged = converged};
}

template <typename S>
[[nodiscard]] SpectrumStatus EigenBatch(const S* const input,
                                        S* const values,
                                        S* const vectors,
                                        const SpectrumPlan& plan,
                                        const u64 batch,
                                        S* const a,
                                        S* const jacobi_vectors,
                                        S* const jacobi_values) {
  const S* const src = input + batch * plan.rows * plan.cols;
  for (u64 row = 0u; row < plan.rows; ++row) {
    for (u64 col = 0u; col < plan.cols; ++col) {
      a[static_cast<std::size_t>(row * plan.rows + col)] =
          src[Index(row, col, plan.rows, plan.cols, plan.layout)];
    }
  }
  JacobiResult eig = JacobiSymmetric(a, jacobi_vectors, jacobi_values,
                                     plan.rows, plan.max_iterations,
                                     plan.fixed_format);
  if (!eig.converged) { return SpectrumStatus::NonConvergence; }
  S* const out = values + batch * plan.rows;
  for (u64 i = 0u; i < plan.rows; ++i) {
    out[i] = jacobi_values[static_cast<std::size_t>(i)];
  }
  if (vectors != nullptr && plan.vector_count != 0u) {
    S* const vec = vectors + batch * plan.vector_count / plan.batch_count;
    const u64 cols =
        plan.vectors == SpectrumVectors::Thin ? plan.rows : plan.rows;
    for (u64 row = 0u; row < plan.rows; ++row) {
      for (u64 col = 0u; col < cols; ++col) {
        vec[Index(row, col, plan.rows, cols, plan.layout)] =
            jacobi_vectors[static_cast<std::size_t>(row * plan.rows + col)];
      }
    }
  }
  return SpectrumStatus::Ok;
}

template <typename S>
void BuildSymmetricAtA(const S* const src, S* const ata,
                       const SpectrumPlan& plan) noexcept {
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
[[nodiscard]] SpectrumStatus SvdBatch(const S* const input,
                                      S* const values,
                                      S* const vectors,
                                      const SpectrumPlan& plan,
                                      const u64 batch,
                                      S* const ata,
                                      S* const jacobi_vectors,
                                      S* const jacobi_values,
                                      u64* const order,
                                      S* const u) {
  const u64 n = plan.cols;
  const S* const src = input + batch * plan.rows * plan.cols;
  BuildSymmetricAtA(src, ata, plan);
  JacobiResult eig = JacobiSymmetric(ata, jacobi_vectors, jacobi_values, n,
                                     plan.max_iterations,
                                     plan.fixed_format);
  if (!eig.converged) { return SpectrumStatus::NonConvergence; }
  for (u64 i = 0u; i < n; ++i) {
    const S value = jacobi_values[static_cast<std::size_t>(i)];
    jacobi_values[static_cast<std::size_t>(i)] =
        value < 0 ? S{0} : factor_reference_detail::SqrtFixed(
                                value, plan.fixed_format);
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
  S* const out = values + batch * width;
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
          Magnitude(sigma) >
              Magnitude(Epsilon<S>(plan.fixed_format))) {
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
              factor_reference_detail::DivFixed(
                  sum, sigma, plan.fixed_format);
        }
      } else {
        u[static_cast<std::size_t>(col * vector_cols + col)] =
            One<S>(plan.fixed_format);
      }
      OrthogonalizeColumn(u, plan.rows, vector_cols, col,
                          plan.fixed_format);
      if (DotColumn(u, plan.rows, vector_cols, col, col,
                    plan.fixed_format) == 0) {
        FillBasisColumn(u, plan.rows, vector_cols, col,
                        plan.fixed_format);
      }
      NormalizeColumn(u, plan.rows, vector_cols, col,
                      plan.fixed_format);
    }
    S* const vec = vectors + batch * plan.vector_count / plan.batch_count;
    for (u64 row = 0u; row < plan.rows; ++row) {
      for (u64 col = 0u; col < vector_cols; ++col) {
        vec[Index(row, col, plan.rows, vector_cols, plan.layout)] =
            u[static_cast<std::size_t>(row * vector_cols + col)];
      }
    }
  }
  return SpectrumStatus::Ok;
}

template <typename S>
[[nodiscard]] SpectrumResult ReferenceSpectrum(const S* const input,
                                               S* const values,
                                               S* const vectors,
                                               u32* const status,
                                               const SpectrumPlan& plan,
                                               S* const matrix,
                                               S* const jacobi_vectors,
                                               S* const jacobi_values,
                                               u64* const order,
                                               S* const u) {
  if (!plan.ok) { return SpectrumResult{.reason = plan.reason}; }
  if (input == nullptr || values == nullptr || status == nullptr ||
      matrix == nullptr || jacobi_vectors == nullptr ||
      jacobi_values == nullptr ||
      (plan.op == SpectrumOp::SVD && order == nullptr) ||
      (plan.vector_count != 0u && vectors == nullptr) ||
      (plan.op == SpectrumOp::SVD && plan.vector_count != 0u &&
       u == nullptr)) {
    return SpectrumResult{.reason = "compute_spectrum_buffer_invalid"};
  }
  SpectrumResult result{.ok = true, .reason = "ok"};
  for (u64 batch = 0u; batch < plan.batch_count; ++batch) {
    const SpectrumStatus batch_status =
        plan.op == SpectrumOp::Eigen
            ? EigenBatch(input, values, vectors, plan, batch, matrix,
                         jacobi_vectors, jacobi_values)
            : SvdBatch(input, values, vectors, plan, batch, matrix,
                       jacobi_vectors, jacobi_values, order, u);
    MarkStatus<S>(status, batch, batch_status);
    if (batch_status != SpectrumStatus::Ok) {
      result = RecordFailure(result, batch, batch_status);
    }
  }
  return result;
}

template <typename S>
[[nodiscard]] SpectrumResult ReferenceSpectrumOwned(
    const S* const input, S* const values, S* const vectors,
    u32* const status, const SpectrumPlan& plan) {
  const u64 n = std::max(plan.rows, plan.cols);
  std::vector<S> matrix(static_cast<std::size_t>(n * n));
  std::vector<S> jacobi_vectors(static_cast<std::size_t>(n * n));
  std::vector<S> jacobi_values(static_cast<std::size_t>(n));
  std::vector<u64> order(static_cast<std::size_t>(n));
  std::vector<S> u(static_cast<std::size_t>(plan.rows * n));
  return ReferenceSpectrum(input, values, vectors, status, plan, matrix.data(),
                           jacobi_vectors.data(), jacobi_values.data(),
                           order.data(), u.data());
}

}  // namespace spectrum_reference_detail

[[nodiscard]] inline SpectrumResult ReferenceSpectrumI32(
    const i32* const input,
    i32* const values,
    i32* const vectors,
    u32* const status,
    const SpectrumPlan& plan) {
  return spectrum_reference_detail::ReferenceSpectrumOwned(
      input, values, vectors, status, plan);
}

[[nodiscard]] inline SpectrumResult ReferenceSpectrumI64(
    const i64* const input,
    i64* const values,
    i64* const vectors,
    u32* const status,
    const SpectrumPlan& plan) {
  return spectrum_reference_detail::ReferenceSpectrumOwned(
      input, values, vectors, status, plan);
}

[[nodiscard]] inline SpectrumResult ReferenceSpectrumScratchI32(
    const i32* const input, i32* const values, i32* const vectors,
    u32* const status, const SpectrumPlan& plan, i32* const matrix,
    i32* const jacobi_vectors, i32* const jacobi_values,
    u64* const order, i32* const u) {
  return spectrum_reference_detail::ReferenceSpectrum(
      input, values, vectors, status, plan, matrix, jacobi_vectors,
      jacobi_values, order, u);
}

[[nodiscard]] inline SpectrumResult ReferenceSpectrumScratchI64(
    const i64* const input, i64* const values, i64* const vectors,
    u32* const status, const SpectrumPlan& plan, i64* const matrix,
    i64* const jacobi_vectors, i64* const jacobi_values,
    u64* const order, i64* const u) {
  return spectrum_reference_detail::ReferenceSpectrum(
      input, values, vectors, status, plan, matrix, jacobi_vectors,
      jacobi_values, order, u);
}

}  // namespace rund::kernel
