#pragma once

#include <kernel/program/compute/spectrum/reference/eigen.hpp>
#include <kernel/program/compute/spectrum/reference/svd.hpp>

#include <algorithm>
#include <vector>

namespace rund::kernel::spectrum_reference_detail {

template <typename S>
[[nodiscard]] SpectrumResult ReferenceSpectrum(
    const S *const input, S *const values, S *const vectors, u32 *const status,
    const SpectrumPlan &plan, S *const matrix, S *const jacobi_vectors,
    S *const jacobi_values, u64 *const order, S *const u) {
  if (!plan.ok) {
    return SpectrumResult{.reason = plan.reason};
  }
  if (input == nullptr || values == nullptr || status == nullptr ||
      matrix == nullptr || jacobi_vectors == nullptr ||
      jacobi_values == nullptr ||
      (plan.op == SpectrumOp::SVD && order == nullptr) ||
      (plan.vector_count != 0u && vectors == nullptr) ||
      (plan.op == SpectrumOp::SVD && plan.vector_count != 0u && u == nullptr)) {
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
    const S *const input, S *const values, S *const vectors, u32 *const status,
    const SpectrumPlan &plan) {
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

} // namespace rund::kernel::spectrum_reference_detail

namespace rund::kernel {

[[nodiscard]] inline SpectrumResult
ReferenceSpectrumI32(const i32 *const input, i32 *const values,
                     i32 *const vectors, u32 *const status,
                     const SpectrumPlan &plan) {
  return spectrum_reference_detail::ReferenceSpectrumOwned(
      input, values, vectors, status, plan);
}

[[nodiscard]] inline SpectrumResult
ReferenceSpectrumI64(const i64 *const input, i64 *const values,
                     i64 *const vectors, u32 *const status,
                     const SpectrumPlan &plan) {
  return spectrum_reference_detail::ReferenceSpectrumOwned(
      input, values, vectors, status, plan);
}

[[nodiscard]] inline SpectrumResult ReferenceSpectrumScratchI32(
    const i32 *const input, i32 *const values, i32 *const vectors,
    u32 *const status, const SpectrumPlan &plan, i32 *const matrix,
    i32 *const jacobi_vectors, i32 *const jacobi_values, u64 *const order,
    i32 *const u) {
  return spectrum_reference_detail::ReferenceSpectrum(
      input, values, vectors, status, plan, matrix, jacobi_vectors,
      jacobi_values, order, u);
}

[[nodiscard]] inline SpectrumResult ReferenceSpectrumScratchI64(
    const i64 *const input, i64 *const values, i64 *const vectors,
    u32 *const status, const SpectrumPlan &plan, i64 *const matrix,
    i64 *const jacobi_vectors, i64 *const jacobi_values, u64 *const order,
    i64 *const u) {
  return spectrum_reference_detail::ReferenceSpectrum(
      input, values, vectors, status, plan, matrix, jacobi_vectors,
      jacobi_values, order, u);
}

} // namespace rund::kernel
