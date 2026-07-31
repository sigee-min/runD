#pragma once

#include <kernel/program/compute/factor/plan.hpp>
#include <kernel/program/compute/fixed/arithmetic.hpp>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace rund::kernel {
namespace factor_reference_detail {

template <typename S>
[[nodiscard]] constexpr S ClampSigned(const i128 value) noexcept {
  const i128 max = (static_cast<i128>(1) << (sizeof(S) * 8u - 1u)) - 1;
  const i128 min =
      -(static_cast<i128>(1) << (sizeof(S) * 8u - 1u));
  if (value > max) { return static_cast<S>(max); }
  if (value < min) { return static_cast<S>(min); }
  return static_cast<S>(value);
}

template <typename S>
[[nodiscard]] constexpr S AddSat(const S lhs, const S rhs) noexcept {
  return ClampSigned<S>(static_cast<i128>(lhs) + rhs);
}

template <typename S>
[[nodiscard]] constexpr S SubSat(const S lhs, const S rhs) noexcept {
  return ClampSigned<S>(static_cast<i128>(lhs) - rhs);
}

template <typename S>
[[nodiscard]] constexpr S MulFixed(const S lhs, const S rhs,
                                   const ComputeFixedFormat format) noexcept {
  return compute_fixed_detail::Mul(lhs, rhs, format);
}

template <typename S>
[[nodiscard]] constexpr S DivFixed(const S lhs, const S rhs,
                                   const ComputeFixedFormat format) noexcept {
  return compute_fixed_detail::Div(lhs, rhs, format);
}

template <typename S>
[[nodiscard]] constexpr S SqrtFixed(const S value,
                                    const ComputeFixedFormat format) noexcept {
  return compute_fixed_detail::Sqrt(value, format);
}

[[nodiscard]] constexpr u64 Index(const u64 row, const u64 col, const u64 rows,
                                  const u64 cols,
                                  const MatrixLayout layout) noexcept {
  return layout == MatrixLayout::RowMajor ? row * cols + col : col * rows + row;
}

template <typename S>
void MarkStatus(u32* const status, const u64 batch,
                const FactorStatus value) noexcept {
  if (status != nullptr) { status[batch] = static_cast<u32>(value); }
}

template <typename S>
[[nodiscard]] FactorResult RecordFailure(FactorResult result, const u64 batch,
                                         const FactorStatus status) noexcept {
  if (result.failed_batches == 0u) {
    result.first_failed_batch = batch;
    result.first_status = status;
  }
  ++result.failed_batches;
  return result;
}

template <typename S>
void CopyBatch(const S* const input, S* const factor, const FactorPlan& plan,
               const u64 batch) noexcept {
  const u64 offset = batch * plan.rows * plan.cols;
  for (u64 index = 0u; index < plan.rows * plan.cols; ++index) {
    factor[offset + index] = input[offset + index];
  }
}

template <typename S>
void SwapRows(S* const a, const FactorPlan& plan, const u64 lhs,
              const u64 rhs) noexcept {
  if (lhs == rhs) { return; }
  for (u64 col = 0u; col < plan.cols; ++col) {
    const u64 li = Index(lhs, col, plan.rows, plan.cols, plan.layout);
    const u64 ri = Index(rhs, col, plan.rows, plan.cols, plan.layout);
    const S tmp = a[li];
    a[li] = a[ri];
    a[ri] = tmp;
  }
}

template <typename S>
[[nodiscard]] FactorStatus FactorLuBatch(
    const S* const input,
    S* const factor,
    u32* const aux,
    const FactorPlan& plan,
    const u64 batch) noexcept {
  CopyBatch(input, factor, plan, batch);
  S* const a = factor + batch * plan.rows * plan.cols;
  u32* const pivots = aux == nullptr ? nullptr : aux + batch * plan.rows;
  for (u64 k = 0u; k < plan.rows; ++k) {
    u64 pivot_row = k;
    if (plan.pivot == PivotOp::Partial) {
      u128 best = 0u;
      for (u64 row = k; row < plan.rows; ++row) {
        const S value = a[Index(row, k, plan.rows, plan.cols, plan.layout)];
        const u128 mag =
            value < 0 ? static_cast<u128>(-static_cast<i128>(value))
                      : static_cast<u128>(value);
        if (mag > best) {
          best = mag;
          pivot_row = row;
        }
      }
    }
    if (pivots != nullptr) { pivots[k] = static_cast<u32>(pivot_row); }
    SwapRows(a, plan, k, pivot_row);
    const S pivot = a[Index(k, k, plan.rows, plan.cols, plan.layout)];
    if (pivot == 0) { return FactorStatus::Singular; }
    for (u64 row = k + 1u; row < plan.rows; ++row) {
      const u64 lk = Index(row, k, plan.rows, plan.cols, plan.layout);
      a[lk] = DivFixed(a[lk], pivot, plan.fixed_format);
      for (u64 col = k + 1u; col < plan.cols; ++col) {
        const u64 rc = Index(row, col, plan.rows, plan.cols, plan.layout);
        const S update =
            MulFixed(a[lk], a[Index(k, col, plan.rows, plan.cols, plan.layout)],
                     plan.fixed_format);
        a[rc] = SubSat(a[rc], update);
      }
    }
  }
  return FactorStatus::Ok;
}

template <typename S>
[[nodiscard]] FactorStatus FactorCholeskyBatch(
    const S* const input,
    S* const factor,
    const FactorPlan& plan,
    const u64 batch) noexcept {
  S* const l = factor + batch * plan.rows * plan.cols;
  for (u64 row = 0u; row < plan.rows; ++row) {
    for (u64 col = 0u; col < plan.cols; ++col) {
      l[Index(row, col, plan.rows, plan.cols, plan.layout)] = 0;
    }
  }
  const S* const a = input + batch * plan.rows * plan.cols;
  for (u64 i = 0u; i < plan.rows; ++i) {
    for (u64 j = 0u; j <= i; ++j) {
      S sum = a[Index(i, j, plan.rows, plan.cols, plan.layout)];
      for (u64 k = 0u; k < j; ++k) {
        sum = SubSat(
            sum,
            MulFixed(l[Index(i, k, plan.rows, plan.cols, plan.layout)],
                     l[Index(j, k, plan.rows, plan.cols, plan.layout)],
                     plan.fixed_format));
      }
      if (i == j) {
        if (sum <= 0) { return FactorStatus::NonSpd; }
        l[Index(i, j, plan.rows, plan.cols, plan.layout)] =
            SqrtFixed(sum, plan.fixed_format);
      } else {
        const S diag = l[Index(j, j, plan.rows, plan.cols, plan.layout)];
        if (diag == 0) { return FactorStatus::NonSpd; }
        l[Index(i, j, plan.rows, plan.cols, plan.layout)] =
            DivFixed(sum, diag, plan.fixed_format);
      }
    }
  }
  return FactorStatus::Ok;
}

template <typename S>
[[nodiscard]] FactorStatus FactorQrWorkspaceBatch(
    const S* const input,
    const FactorPlan& plan,
    const u64 batch,
    S* const q,
    S* const r,
    S* const v) {
  const u64 rows = plan.rows;
  const u64 cols = plan.cols;
  if (q == nullptr || r == nullptr || v == nullptr) {
    return FactorStatus::InvalidScaling;
  }
  for (u64 col = 0u; col < cols; ++col) {
    for (u64 row = 0u; row < rows; ++row) {
      v[static_cast<std::size_t>(row)] = input[
          batch * rows * cols + Index(row, col, rows, cols, plan.layout)];
    }
    for (u64 j = 0u; j < col; ++j) {
      S dot = 0;
      for (u64 row = 0u; row < rows; ++row) {
        dot = AddSat(
            dot, MulFixed(q[static_cast<std::size_t>(row * cols + j)],
                          v[static_cast<std::size_t>(row)],
                          plan.fixed_format));
      }
      r[static_cast<std::size_t>(j * cols + col)] = dot;
      for (u64 row = 0u; row < rows; ++row) {
        v[static_cast<std::size_t>(row)] = SubSat(
            v[static_cast<std::size_t>(row)],
            MulFixed(dot, q[static_cast<std::size_t>(row * cols + j)],
                     plan.fixed_format));
      }
    }
    S norm_squared = 0;
    for (u64 row = 0u; row < rows; ++row) {
      norm_squared = AddSat(
          norm_squared,
          MulFixed(v[static_cast<std::size_t>(row)],
                   v[static_cast<std::size_t>(row)],
                   plan.fixed_format));
    }
    const S norm = SqrtFixed(norm_squared, plan.fixed_format);
    if (norm == 0) { return FactorStatus::Singular; }
    r[static_cast<std::size_t>(col * cols + col)] = norm;
    for (u64 row = 0u; row < rows; ++row) {
      q[static_cast<std::size_t>(row * cols + col)] =
          DivFixed(v[static_cast<std::size_t>(row)], norm,
                   plan.fixed_format);
    }
  }
  return FactorStatus::Ok;
}

template <typename S>
[[nodiscard]] FactorStatus FactorQrBatch(
    const S* const input,
    S* const factor,
    const FactorPlan& plan,
    const u64 batch,
    S* const q,
    S* const r,
    S* const v) {
  const FactorStatus status =
      FactorQrWorkspaceBatch(input, plan, batch, q, r, v);
  if (status != FactorStatus::Ok) { return status; }
  const u64 rows = plan.rows;
  const u64 cols = plan.cols;
  S* const out = factor + batch * plan.factor_count / plan.batch_count;
  for (u64 row = 0u; row < rows; ++row) {
    for (u64 col = 0u; col < cols; ++col) {
      out[Index(row, col, rows, cols, plan.layout)] =
          q[static_cast<std::size_t>(row * cols + col)];
    }
  }
  if (plan.output == FactorOutput::Separate) {
    S* const r_out = out + rows * cols;
    for (u64 row = 0u; row < cols; ++row) {
      for (u64 col = 0u; col < row; ++col) {
        r_out[Index(row, col, cols, cols, plan.layout)] = S{0};
      }
      for (u64 col = row; col < cols; ++col) {
        r_out[Index(row, col, cols, cols, plan.layout)] =
            r[static_cast<std::size_t>(row * cols + col)];
      }
    }
  }
  return FactorStatus::Ok;
}

template <typename S>
[[nodiscard]] FactorResult ReferenceFactor(const S* const input,
                                           S* const factor,
                                           u32* const aux,
                                           u32* const status,
                                           const FactorPlan& plan,
                                           S* const q,
                                           S* const r,
                                           S* const v) {
  if (!plan.ok) { return FactorResult{.reason = plan.reason}; }
  if (input == nullptr || factor == nullptr || status == nullptr ||
      (plan.op == FactorOp::LU && aux == nullptr)) {
    return FactorResult{.reason = "compute_factor_buffer_invalid"};
  }
  FactorResult result{.ok = true, .reason = "ok"};
  for (u64 batch = 0u; batch < plan.batch_count; ++batch) {
    FactorStatus batch_status = FactorStatus::Ok;
    if (plan.op == FactorOp::LU) {
      batch_status = FactorLuBatch(input, factor, aux, plan, batch);
    } else if (plan.op == FactorOp::Cholesky) {
      batch_status = FactorCholeskyBatch(input, factor, plan, batch);
    } else {
      batch_status = FactorQrBatch(input, factor, plan, batch, q, r, v);
    }
    MarkStatus<S>(status, batch, batch_status);
    if (batch_status != FactorStatus::Ok) {
      result = RecordFailure<S>(result, batch, batch_status);
    }
  }
  return result;
}

template <typename S>
[[nodiscard]] FactorResult ReferenceFactorOwned(const S* const input,
                                                S* const factor,
                                                u32* const aux,
                                                u32* const status,
                                                const FactorPlan& plan) {
  std::vector<S> q;
  std::vector<S> r;
  std::vector<S> v;
  if (plan.op == FactorOp::QR) {
    q.resize(static_cast<std::size_t>(plan.rows * plan.cols));
    r.resize(static_cast<std::size_t>(plan.cols * plan.cols));
    v.resize(static_cast<std::size_t>(plan.rows));
  }
  return ReferenceFactor(input, factor, aux, status, plan, q.data(), r.data(),
                         v.data());
}

}  // namespace factor_reference_detail

[[nodiscard]] inline FactorResult ReferenceFactorI32(
    const i32* const input,
    i32* const factor,
    u32* const aux,
    u32* const status,
    const FactorPlan& plan) {
  return factor_reference_detail::ReferenceFactorOwned(input, factor, aux,
                                                       status, plan);
}

[[nodiscard]] inline FactorResult ReferenceFactorI64(
    const i64* const input,
    i64* const factor,
    u32* const aux,
    u32* const status,
    const FactorPlan& plan) {
  return factor_reference_detail::ReferenceFactorOwned(input, factor, aux,
                                                       status, plan);
}

[[nodiscard]] inline FactorResult ReferenceFactorScratchI32(
    const i32* const input, i32* const factor, u32* const aux,
    u32* const status, const FactorPlan& plan, i32* const q,
    i32* const r, i32* const v) {
  return factor_reference_detail::ReferenceFactor(input, factor, aux, status,
                                                   plan, q, r, v);
}

[[nodiscard]] inline FactorResult ReferenceFactorScratchI64(
    const i64* const input, i64* const factor, u32* const aux,
    u32* const status, const FactorPlan& plan, i64* const q,
    i64* const r, i64* const v) {
  return factor_reference_detail::ReferenceFactor(input, factor, aux, status,
                                                   plan, q, r, v);
}

}  // namespace rund::kernel
