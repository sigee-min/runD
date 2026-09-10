#pragma once

#include <kernel/program/compute/solve/reference/cholesky.hpp>
#include <kernel/program/compute/solve/reference/lu.hpp>
#include <kernel/program/compute/solve/reference/qr.hpp>

#include <cstddef>
#include <vector>

namespace rund::kernel {
namespace solve_reference_detail {

template <typename S>
[[nodiscard]] SolveResult
ReferenceSolve(const S *const matrix_or_factor, const u32 *const aux,
               const S *const rhs, S *const output, u32 *const status,
               const SolvePlan &plan, S *const factor_storage,
               u32 *const aux_storage, S *const y, S *const factor_q,
               S *const factor_r, S *const factor_v) {
  if (!plan.ok) {
    return SolveResult{.reason = plan.reason};
  }
  if (matrix_or_factor == nullptr || rhs == nullptr || output == nullptr ||
      status == nullptr ||
      (plan.input == SolveInput::Factor && plan.factor == FactorOp::LU &&
       aux == nullptr)) {
    return SolveResult{.reason = "compute_solve_buffer_invalid"};
  }
  SolveResult result{.ok = true, .reason = "ok"};
  if (plan.input == SolveInput::Matrix && plan.factor != FactorOp::QR &&
      factor_storage == nullptr) {
    return SolveResult{.reason = "compute_solve_buffer_invalid"};
  }
  if (plan.input == SolveInput::Matrix && plan.factor == FactorOp::LU &&
      aux_storage == nullptr) {
    return SolveResult{.reason = "compute_solve_buffer_invalid"};
  }

  for (u64 batch = 0u; batch < plan.batch_count; ++batch) {
    const S *factor = matrix_or_factor;
    const u32 *pivots = aux;
    SolveStatus batch_status = SolveStatus::Ok;
    if (plan.input == SolveInput::Matrix) {
      const u64 factor_stride = plan.factor_count / plan.batch_count;
      FactorPlan factor_plan{
          .op = plan.factor,
          .layout = plan.layout,
          .output = plan.factor == FactorOp::QR ? FactorOutput::Separate
                                                : FactorOutput::Packed,
          .pivot = plan.pivot,
          .rows = plan.rows,
          .cols = plan.rows,
          .batch_count = 1u,
          .input_count = plan.rows * plan.rows,
          .factor_count = factor_stride,
          .aux_count = plan.factor == FactorOp::LU ? plan.rows : 0u,
          .status_count = 1u,
          .workspace_bytes = factor_stride * plan.element_bytes,
          .element_bytes = plan.element_bytes,
          .fixed_format = plan.fixed_format,
          .pass_count = 1u,
          .ok = true,
          .reason = "ok",
      };
      u32 factor_status = 0u;
      const S *const matrix_batch =
          matrix_or_factor + batch * plan.rows * plan.rows;
      S *const factor_batch = plan.factor == FactorOp::QR
                                  ? nullptr
                                  : factor_storage + batch * factor_stride;
      u32 *const aux_batch =
          aux_storage == nullptr ? nullptr : aux_storage + batch * plan.rows;
      if (plan.factor == FactorOp::QR) {
        factor_status =
            static_cast<u32>(factor_reference_detail::FactorQrWorkspaceBatch(
                matrix_batch, factor_plan, 0u, factor_q, factor_r, factor_v));
      } else {
        const FactorResult factored = factor_reference_detail::ReferenceFactor(
            matrix_batch, factor_batch, aux_batch, &factor_status, factor_plan,
            factor_q, factor_r, factor_v);
        if (!factored.ok) {
          return SolveResult{.reason = factored.reason};
        }
      }
      batch_status = FromFactorStatus(static_cast<FactorStatus>(factor_status));
      factor = factor_storage;
      pivots = aux_storage;
    }
    if (batch_status == SolveStatus::Ok) {
      if (plan.factor == FactorOp::LU) {
        batch_status = SolveLuBatch(factor, pivots, rhs, output, plan, batch);
      } else if (plan.factor == FactorOp::Cholesky) {
        batch_status = SolveCholeskyBatch(factor, rhs, output, plan, batch);
      } else if (plan.input == SolveInput::Matrix) {
        batch_status =
            SolveQrWorkspaceBatch(factor_q, factor_r, MatrixLayout::RowMajor,
                                  rhs, output, plan, batch, y);
      } else {
        batch_status = SolveQrBatch(factor, rhs, output, plan, batch, y);
      }
    }
    MarkStatus<S>(status, batch, batch_status);
    if (batch_status != SolveStatus::Ok) {
      result = RecordFailure(result, batch, batch_status);
    }
  }
  return result;
}

template <typename S>
[[nodiscard]] SolveResult
ReferenceSolveOwned(const S *const matrix_or_factor, const u32 *const aux,
                    const S *const rhs, S *const output, u32 *const status,
                    const SolvePlan &plan) {
  std::vector<S> factor_storage;
  std::vector<u32> aux_storage;
  std::vector<S> y;
  std::vector<S> factor_q;
  std::vector<S> factor_r;
  std::vector<S> factor_v;
  if (plan.input == SolveInput::Matrix && plan.factor != FactorOp::QR) {
    factor_storage.resize(static_cast<std::size_t>(plan.factor_count));
    if (plan.factor == FactorOp::LU) {
      aux_storage.resize(static_cast<std::size_t>(plan.aux_count));
    }
  }
  if (plan.input == SolveInput::Matrix && plan.factor == FactorOp::QR) {
    factor_q.resize(static_cast<std::size_t>(plan.rows * plan.rows));
    factor_r.resize(static_cast<std::size_t>(plan.rows * plan.rows));
    factor_v.resize(static_cast<std::size_t>(plan.rows));
  }
  if (plan.factor == FactorOp::QR) {
    y.resize(static_cast<std::size_t>(plan.rows * plan.rhs_cols));
  }
  return ReferenceSolve(matrix_or_factor, aux, rhs, output, status, plan,
                        factor_storage.data(), aux_storage.data(), y.data(),
                        factor_q.data(), factor_r.data(), factor_v.data());
}

} // namespace solve_reference_detail

[[nodiscard]] inline SolveResult
ReferenceSolveI32(const i32 *const matrix_or_factor, const u32 *const aux,
                  const i32 *const rhs, i32 *const output, u32 *const status,
                  const SolvePlan &plan) {
  return solve_reference_detail::ReferenceSolveOwned(matrix_or_factor, aux, rhs,
                                                     output, status, plan);
}

[[nodiscard]] inline SolveResult
ReferenceSolveI64(const i64 *const matrix_or_factor, const u32 *const aux,
                  const i64 *const rhs, i64 *const output, u32 *const status,
                  const SolvePlan &plan) {
  return solve_reference_detail::ReferenceSolveOwned(matrix_or_factor, aux, rhs,
                                                     output, status, plan);
}

[[nodiscard]] inline SolveResult ReferenceSolveScratchI32(
    const i32 *const matrix_or_factor, const u32 *const aux,
    const i32 *const rhs, i32 *const output, u32 *const status,
    const SolvePlan &plan, i32 *const factor_storage, u32 *const aux_storage,
    i32 *const y, i32 *const factor_q, i32 *const factor_r,
    i32 *const factor_v) {
  return solve_reference_detail::ReferenceSolve(
      matrix_or_factor, aux, rhs, output, status, plan, factor_storage,
      aux_storage, y, factor_q, factor_r, factor_v);
}

[[nodiscard]] inline SolveResult ReferenceSolveScratchI64(
    const i64 *const matrix_or_factor, const u32 *const aux,
    const i64 *const rhs, i64 *const output, u32 *const status,
    const SolvePlan &plan, i64 *const factor_storage, u32 *const aux_storage,
    i64 *const y, i64 *const factor_q, i64 *const factor_r,
    i64 *const factor_v) {
  return solve_reference_detail::ReferenceSolve(
      matrix_or_factor, aux, rhs, output, status, plan, factor_storage,
      aux_storage, y, factor_q, factor_r, factor_v);
}

} // namespace rund::kernel
