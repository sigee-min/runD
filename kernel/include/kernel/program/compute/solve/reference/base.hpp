#pragma once

#include <kernel/program/compute/factor/reference.hpp>
#include <kernel/program/compute/solve/plan.hpp>

namespace rund::kernel {
namespace solve_reference_detail {

template <typename S>
[[nodiscard]] constexpr S MulFixed(const S lhs, const S rhs,
                                   const ComputeFixedFormat format) noexcept {
  return factor_reference_detail::MulFixed(lhs, rhs, format);
}

template <typename S>
[[nodiscard]] constexpr S DivFixed(const S lhs, const S rhs,
                                   const ComputeFixedFormat format) noexcept {
  return factor_reference_detail::DivFixed(lhs, rhs, format);
}

template <typename S>
[[nodiscard]] constexpr S SubSat(const S lhs, const S rhs) noexcept {
  return factor_reference_detail::SubSat(lhs, rhs);
}

[[nodiscard]] constexpr u64 Index(const u64 row, const u64 col, const u64 rows,
                                  const u64 cols,
                                  const MatrixLayout layout) noexcept {
  return factor_reference_detail::Index(row, col, rows, cols, layout);
}

[[nodiscard]] constexpr SolveStatus
FromFactorStatus(const FactorStatus status) noexcept {
  switch (status) {
  case FactorStatus::Ok:
    return SolveStatus::Ok;
  case FactorStatus::NonSpd:
    return SolveStatus::NonSpd;
  case FactorStatus::PivotUnderflow:
    return SolveStatus::PivotUnderflow;
  case FactorStatus::InvalidScaling:
    return SolveStatus::InvalidScaling;
  case FactorStatus::Singular:
  default:
    return SolveStatus::Singular;
  }
}

template <typename S>
void MarkStatus(u32 *const status, const u64 batch,
                const SolveStatus value) noexcept {
  if (status != nullptr) {
    status[batch] = static_cast<u32>(value);
  }
}

[[nodiscard]] inline SolveResult
RecordFailure(SolveResult result, const u64 batch,
              const SolveStatus status) noexcept {
  if (result.failed_batches == 0u) {
    result.first_failed_batch = batch;
    result.first_status = status;
  }
  ++result.failed_batches;
  return result;
}

} // namespace solve_reference_detail
} // namespace rund::kernel
