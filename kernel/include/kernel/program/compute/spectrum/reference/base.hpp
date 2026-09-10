#pragma once

#include <kernel/program/compute/factor/reference.hpp>
#include <kernel/program/compute/spectrum/plan.hpp>

#include <cstddef>
#include <limits>
#include <type_traits>

namespace rund::kernel::spectrum_reference_detail {

[[nodiscard]] constexpr u64 Index(const u64 row, const u64 col, const u64 rows,
                                  const u64 cols,
                                  const MatrixLayout layout) noexcept {
  return factor_reference_detail::Index(row, col, rows, cols, layout);
}

template <typename S>
void MarkStatus(u32 *const status, const u64 batch,
                const SpectrumStatus value) noexcept {
  if (status != nullptr) {
    status[batch] = static_cast<u32>(value);
  }
}

[[nodiscard]] inline SpectrumResult
RecordFailure(SpectrumResult result, const u64 batch,
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
[[nodiscard]] constexpr S
NormalizeDivide(const S value, const S norm,
                const ComputeFixedFormat format) noexcept {
  if (Magnitude(value) == Magnitude(norm)) {
    return (value < 0) != (norm < 0) ? -One<S>(format) : One<S>(format);
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
[[nodiscard]] inline S DotColumn(const S *const matrix, const u64 rows,
                                 const u64 cols, const u64 lhs, const u64 rhs,
                                 const ComputeFixedFormat format) noexcept {
  S dot = 0;
  for (u64 row = 0u; row < rows; ++row) {
    dot = factor_reference_detail::AddSat(
        dot, factor_reference_detail::MulFixed(
                 matrix[static_cast<std::size_t>(row * cols + lhs)],
                 matrix[static_cast<std::size_t>(row * cols + rhs)], format));
  }
  return dot;
}

template <typename S>
inline void NormalizeColumn(S *const matrix, const u64 rows, const u64 cols,
                            const u64 col,
                            const ComputeFixedFormat format) noexcept {
  const S squared = DotColumn(matrix, rows, cols, col, col, format);
  const S norm = factor_reference_detail::SqrtFixed(squared, format);
  if (norm == 0) {
    return;
  }
  for (u64 row = 0u; row < rows; ++row) {
    S &value = matrix[static_cast<std::size_t>(row * cols + col)];
    value = NormalizeDivide(value, norm, format);
  }
}

template <typename S>
inline void OrthogonalizeColumn(S *const matrix, const u64 rows,
                                const u64 cols, const u64 col,
                                const ComputeFixedFormat format) noexcept {
  for (u64 prev = 0u; prev < col; ++prev) {
    const S dot = DotColumn(matrix, rows, cols, prev, col, format);
    for (u64 row = 0u; row < rows; ++row) {
      S &value = matrix[static_cast<std::size_t>(row * cols + col)];
      value = factor_reference_detail::SubSat(
          value, factor_reference_detail::MulFixed(
                     dot, matrix[static_cast<std::size_t>(row * cols + prev)],
                     format));
    }
  }
}

template <typename S>
inline void FillBasisColumn(S *const matrix, const u64 rows, const u64 cols,
                            const u64 col,
                            const ComputeFixedFormat format) noexcept {
  for (u64 row = 0u; row < rows; ++row) {
    matrix[static_cast<std::size_t>(row * cols + col)] =
        row == (col % rows) ? One<S>(format) : S{0};
  }
}

} // namespace rund::kernel::spectrum_reference_detail
