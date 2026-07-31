#pragma once

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/factor/model.hpp>

namespace rund::kernel {
namespace factor_plan_detail {

constexpr u64 kMaxDimension = 16u;

[[nodiscard]] constexpr FactorPlan Reject(const FactorDesc &desc,
                                          const char *const reason) noexcept {
  return FactorPlan{
      .op = desc.op,
      .layout = desc.layout,
      .output = desc.output,
      .pivot = desc.pivot,
      .rows = desc.rows,
      .cols = desc.cols,
      .batch_count = desc.batch_count,
      .element_bytes = desc.element_bytes,
      .fixed_format = desc.fixed_format,
      .reason = reason,
  };
}

} // namespace factor_plan_detail

[[nodiscard]] constexpr FactorPlan PlanFactor(const FactorDesc &desc) noexcept {
  if (desc.op != FactorOp::LU && desc.op != FactorOp::QR &&
      desc.op != FactorOp::Cholesky) {
    return factor_plan_detail::Reject(desc, "compute_factor_op_unsupported");
  }
  if (desc.layout != MatrixLayout::RowMajor &&
      desc.layout != MatrixLayout::ColumnMajor) {
    return factor_plan_detail::Reject(desc,
                                      "compute_factor_layout_unsupported");
  }
  if (desc.output != FactorOutput::Packed &&
      desc.output != FactorOutput::Separate) {
    return factor_plan_detail::Reject(desc,
                                      "compute_factor_output_unsupported");
  }
  if (desc.pivot != PivotOp::None && desc.pivot != PivotOp::Partial) {
    return factor_plan_detail::Reject(desc, "compute_factor_pivot_unsupported");
  }
  if (desc.element_bytes != 4u && desc.element_bytes != 8u) {
    return factor_plan_detail::Reject(desc,
                                      "compute_factor_element_unsupported");
  }
  if (!ComputePrimitiveFixedFormatValid(desc.element_bytes, desc.fixed_format,
                                        ComputeApproximation::Deterministic)) {
    return factor_plan_detail::Reject(
        desc, "compute_factor_numeric_policy_unsupported");
  }
  if (desc.rows == 0u || desc.cols == 0u || desc.batch_count == 0u) {
    return factor_plan_detail::Reject(desc, "compute_factor_shape_zero");
  }
  if (desc.rows > factor_plan_detail::kMaxDimension ||
      desc.cols > factor_plan_detail::kMaxDimension) {
    return factor_plan_detail::Reject(desc, "compute_factor_shape_dimension");
  }
  if ((desc.op == FactorOp::LU || desc.op == FactorOp::Cholesky) &&
      desc.rows != desc.cols) {
    return factor_plan_detail::Reject(desc, "compute_factor_shape_square");
  }
  if (desc.op == FactorOp::Cholesky && desc.pivot != PivotOp::None) {
    return factor_plan_detail::Reject(desc, "compute_factor_pivot_unsupported");
  }

  u64 input_count = 0u;
  if (!checked::mul(desc.rows, desc.cols, desc.batch_count, input_count)) {
    return factor_plan_detail::Reject(desc, "compute_factor_shape_overflow");
  }
  u64 factor_count = input_count;
  if (desc.output == FactorOutput::Separate && desc.op == FactorOp::QR) {
    u64 q_count = 0u;
    u64 r_count = 0u;
    if (!checked::mul(desc.rows, desc.cols, desc.batch_count, q_count) ||
        !checked::mul(desc.cols, desc.cols, desc.batch_count, r_count) ||
        !checked::add(q_count, r_count)) {
      return factor_plan_detail::Reject(desc, "compute_factor_shape_overflow");
    }
    factor_count = q_count + r_count;
  }
  const u64 aux_count =
      desc.op == FactorOp::LU ? desc.rows * desc.batch_count : 0u;
  const u64 status_count = desc.batch_count;
  u64 workspace_bytes = 0u;
  if (!checked::mul(input_count, desc.element_bytes)) {
    return factor_plan_detail::Reject(desc, "compute_factor_shape_overflow");
  }
  workspace_bytes = input_count * desc.element_bytes;

  return FactorPlan{
      .op = desc.op,
      .layout = desc.layout,
      .output = desc.output,
      .pivot = desc.pivot,
      .rows = desc.rows,
      .cols = desc.cols,
      .batch_count = desc.batch_count,
      .input_count = input_count,
      .factor_count = factor_count,
      .aux_count = aux_count,
      .status_count = status_count,
      .workspace_bytes = workspace_bytes,
      .element_bytes = desc.element_bytes,
      .fixed_format = desc.fixed_format,
      .pass_count = 1u,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] constexpr bool
FactorPlanMatchesDesc(const FactorDesc &desc, const FactorPlan &plan) noexcept {
  return plan.ok && desc.op == plan.op && desc.layout == plan.layout &&
         desc.output == plan.output && desc.pivot == plan.pivot &&
         desc.rows == plan.rows && desc.cols == plan.cols &&
         desc.batch_count == plan.batch_count &&
         desc.element_bytes == plan.element_bytes &&
         desc.fixed_format == plan.fixed_format && plan.pass_count == 1u;
}

} // namespace rund::kernel
