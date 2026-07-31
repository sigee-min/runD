#pragma once

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/solve/model.hpp>

namespace rund::kernel {
namespace solve_plan_detail {

constexpr u64 kMaxDimension = 16u;

[[nodiscard]] constexpr SolvePlan Reject(const SolveDesc &desc,
                                         const char *const reason) noexcept {
  return SolvePlan{
      .op = desc.op,
      .input = desc.input,
      .factor = desc.factor,
      .layout = desc.layout,
      .pivot = desc.pivot,
      .rows = desc.rows,
      .rhs_cols = desc.rhs_cols,
      .batch_count = desc.batch_count,
      .element_bytes = desc.element_bytes,
      .fixed_format = desc.fixed_format,
      .reason = reason,
  };
}

} // namespace solve_plan_detail

[[nodiscard]] constexpr SolvePlan PlanSolve(const SolveDesc &desc) noexcept {
  if (desc.op != SolveOp::Linear) {
    return solve_plan_detail::Reject(desc, "compute_solve_op_unsupported");
  }
  if (desc.input != SolveInput::Matrix && desc.input != SolveInput::Factor) {
    return solve_plan_detail::Reject(desc, "compute_solve_input_unsupported");
  }
  if (desc.factor != FactorOp::LU && desc.factor != FactorOp::QR &&
      desc.factor != FactorOp::Cholesky) {
    return solve_plan_detail::Reject(desc, "compute_solve_factor_unsupported");
  }
  if (desc.layout != MatrixLayout::RowMajor &&
      desc.layout != MatrixLayout::ColumnMajor) {
    return solve_plan_detail::Reject(desc, "compute_solve_layout_unsupported");
  }
  if (desc.pivot != PivotOp::None && desc.pivot != PivotOp::Partial) {
    return solve_plan_detail::Reject(desc, "compute_solve_pivot_unsupported");
  }
  if (desc.element_bytes != 4u && desc.element_bytes != 8u) {
    return solve_plan_detail::Reject(desc, "compute_solve_element_unsupported");
  }
  if (!ComputePrimitiveFixedFormatValid(desc.element_bytes, desc.fixed_format,
                                        ComputeApproximation::Deterministic)) {
    return solve_plan_detail::Reject(
        desc, "compute_solve_numeric_policy_unsupported");
  }
  if (desc.rows == 0u || desc.rhs_cols == 0u || desc.batch_count == 0u) {
    return solve_plan_detail::Reject(desc, "compute_solve_shape_zero");
  }
  if (desc.rows > solve_plan_detail::kMaxDimension ||
      desc.rhs_cols > solve_plan_detail::kMaxDimension) {
    return solve_plan_detail::Reject(desc, "compute_solve_shape_dimension");
  }
  if (desc.factor == FactorOp::Cholesky && desc.pivot != PivotOp::None) {
    return solve_plan_detail::Reject(desc, "compute_solve_pivot_unsupported");
  }
  if (desc.factor == FactorOp::QR && desc.pivot != PivotOp::None) {
    return solve_plan_detail::Reject(desc, "compute_solve_pivot_unsupported");
  }

  u64 matrix_count = 0u;
  u64 rhs_count = 0u;
  if (!checked::mul(desc.rows, desc.rows, desc.batch_count, matrix_count) ||
      !checked::mul(desc.rows, desc.rhs_cols, desc.batch_count, rhs_count)) {
    return solve_plan_detail::Reject(desc, "compute_solve_shape_overflow");
  }
  u64 factor_count = matrix_count;
  if (desc.factor == FactorOp::QR) {
    if (!checked::mul(matrix_count, 2u)) {
      return solve_plan_detail::Reject(desc, "compute_solve_shape_overflow");
    }
    factor_count = matrix_count * 2u;
  }
  const u64 aux_count =
      desc.factor == FactorOp::LU ? desc.rows * desc.batch_count : 0u;
  const u64 status_count = desc.batch_count;
  if (!checked::mul(factor_count, desc.element_bytes)) {
    return solve_plan_detail::Reject(desc, "compute_solve_shape_overflow");
  }

  return SolvePlan{
      .op = desc.op,
      .input = desc.input,
      .factor = desc.factor,
      .layout = desc.layout,
      .pivot = desc.pivot,
      .rows = desc.rows,
      .rhs_cols = desc.rhs_cols,
      .batch_count = desc.batch_count,
      .matrix_count = desc.input == SolveInput::Matrix ? matrix_count : 0u,
      .factor_count = factor_count,
      .rhs_count = rhs_count,
      .output_count = rhs_count,
      .aux_count = aux_count,
      .status_count = status_count,
      .workspace_bytes = factor_count * desc.element_bytes,
      .element_bytes = desc.element_bytes,
      .fixed_format = desc.fixed_format,
      .pass_count = 1u,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] constexpr bool
SolvePlanMatchesDesc(const SolveDesc &desc, const SolvePlan &plan) noexcept {
  return plan.ok && desc.op == plan.op && desc.input == plan.input &&
         desc.factor == plan.factor && desc.layout == plan.layout &&
         desc.pivot == plan.pivot && desc.rows == plan.rows &&
         desc.rhs_cols == plan.rhs_cols &&
         desc.batch_count == plan.batch_count &&
         desc.element_bytes == plan.element_bytes &&
         desc.fixed_format == plan.fixed_format && plan.pass_count == 1u;
}

} // namespace rund::kernel
