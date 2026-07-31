#pragma once

#include <kernel/program/compute/matrix/plan.hpp>
#include <kernel/program/compute/fixed/arithmetic.hpp>

#include <bit>
#include <type_traits>

namespace rund::kernel {
namespace matrix_reference_detail {

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
[[nodiscard]] constexpr S MulFixed(
    const S lhs, const S rhs, const ComputeFixedFormat format) noexcept {
  return compute_fixed_detail::Mul(lhs, rhs, format);
}

template <typename S>
[[nodiscard]] constexpr S AddWrap(const S lhs, const S rhs) noexcept {
  using U = std::make_unsigned_t<S>;
  const U value = static_cast<U>(lhs) + static_cast<U>(rhs);
  return std::bit_cast<S>(value);
}

template <typename S>
[[nodiscard]] constexpr S MulWrap(const S lhs, const S rhs) noexcept {
  using U = std::make_unsigned_t<S>;
  const U value = static_cast<U>(lhs) * static_cast<U>(rhs);
  return std::bit_cast<S>(value);
}

[[nodiscard]] constexpr u64 RowMajorIndex(const u64 row, const u64 col,
                                          const u64 rows,
                                          const u64 cols,
                                          const MatrixLayout layout) noexcept {
  return layout == MatrixLayout::RowMajor ? row * cols + col : col * rows + row;
}

template <typename S>
[[nodiscard]] MatrixResult ReferenceMatrixMul(const S* const left,
                                              const S* const right,
                                              S* const output,
                                              const MatrixPlan& plan) noexcept {
  if (!plan.ok) { return MatrixResult{.reason = plan.reason}; }
  if (left == nullptr || right == nullptr || output == nullptr) {
    return MatrixResult{.output_count = plan.output_count,
                        .reason = "compute_matrix_buffer_invalid"};
  }
  for (u64 batch = 0u; batch < plan.batch_count; ++batch) {
    const S* const a = left + batch * plan.rows * plan.inner;
    const S* const b = right + batch * plan.inner * plan.cols;
    S* const c = output + batch * plan.rows * plan.cols;
    for (u64 row = 0u; row < plan.rows; ++row) {
      for (u64 col = 0u; col < plan.cols; ++col) {
        S acc = 0;
        for (u64 inner = 0u; inner < plan.inner; ++inner) {
          const S av = a[RowMajorIndex(row, inner, plan.rows, plan.inner,
                                       plan.layout)];
          const S bv = b[RowMajorIndex(inner, col, plan.inner, plan.cols,
                                       plan.layout)];
          acc = plan.arithmetic == MatrixArithmetic::Fixed
                    ? AddSat(acc, MulFixed(av, bv, plan.fixed_format))
                    : AddWrap(acc, MulWrap(av, bv));
        }
        c[RowMajorIndex(row, col, plan.rows, plan.cols, plan.layout)] =
            acc;
      }
    }
  }
  return MatrixResult{.output_count = plan.output_count,
                      .ok = true,
                      .reason = "ok"};
}

template <typename S>
[[nodiscard]] MatrixResult ReferenceMatrixTranspose(
    const S* const input,
    S* const output,
    const MatrixPlan& plan) noexcept {
  if (!plan.ok) { return MatrixResult{.reason = plan.reason}; }
  if (input == nullptr || output == nullptr) {
    return MatrixResult{.output_count = plan.output_count,
                        .reason = "compute_matrix_buffer_invalid"};
  }
  for (u64 batch = 0u; batch < plan.batch_count; ++batch) {
    const S* const src = input + batch * plan.rows * plan.cols;
    S* const dst = output + batch * plan.rows * plan.cols;
    for (u64 row = 0u; row < plan.rows; ++row) {
      for (u64 col = 0u; col < plan.cols; ++col) {
        const u64 src_index =
            RowMajorIndex(row, col, plan.rows, plan.cols, plan.layout);
        const u64 dst_index =
            RowMajorIndex(col, row, plan.cols, plan.rows, plan.layout);
        dst[dst_index] = src[src_index];
      }
    }
  }
  return MatrixResult{.output_count = plan.output_count,
                      .ok = true,
                      .reason = "ok"};
}

}  // namespace matrix_reference_detail

[[nodiscard]] inline MatrixResult ReferenceMatrixMulI32(
    const i32* const left,
    const i32* const right,
    i32* const output,
    const MatrixPlan& plan) noexcept {
  return matrix_reference_detail::ReferenceMatrixMul(left, right, output, plan);
}

[[nodiscard]] inline MatrixResult ReferenceMatrixMulI64(
    const i64* const left,
    const i64* const right,
    i64* const output,
    const MatrixPlan& plan) noexcept {
  return matrix_reference_detail::ReferenceMatrixMul(left, right, output, plan);
}

[[nodiscard]] inline MatrixResult ReferenceMatrixTransposeI32(
    const i32* const input,
    i32* const output,
    const MatrixPlan& plan) noexcept {
  return matrix_reference_detail::ReferenceMatrixTranspose(input, output, plan);
}

[[nodiscard]] inline MatrixResult ReferenceMatrixTransposeI64(
    const i64* const input,
    i64* const output,
    const MatrixPlan& plan) noexcept {
  return matrix_reference_detail::ReferenceMatrixTranspose(input, output, plan);
}

}  // namespace rund::kernel
