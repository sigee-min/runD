#pragma once

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/matrix/model.hpp>

namespace rund::kernel {
namespace matrix_plan_detail {

[[nodiscard]] constexpr MatrixPlan Reject(const MatrixDesc &desc,
                                          const char *const reason) noexcept {
  return MatrixPlan{
      .op = desc.op,
      .layout = desc.layout,
      .arithmetic = desc.arithmetic,
      .rows = desc.rows,
      .cols = desc.cols,
      .inner = desc.inner,
      .batch_count = desc.batch_count,
      .element_bytes = desc.element_bytes,
      .fixed_format = desc.fixed_format,
      .reason = reason,
  };
}

} // namespace matrix_plan_detail

[[nodiscard]] constexpr MatrixPlan PlanMatrix(const MatrixDesc &desc) noexcept {
  if (desc.op != MatrixOp::Mul && desc.op != MatrixOp::Transpose &&
      desc.op != MatrixOp::BatchMul) {
    return matrix_plan_detail::Reject(desc, "compute_matrix_op_unsupported");
  }
  if (desc.layout != MatrixLayout::RowMajor &&
      desc.layout != MatrixLayout::ColumnMajor) {
    return matrix_plan_detail::Reject(desc,
                                      "compute_matrix_layout_unsupported");
  }
  if (desc.arithmetic != MatrixArithmetic::Fixed &&
      desc.arithmetic != MatrixArithmetic::SignedWrap &&
      desc.arithmetic != MatrixArithmetic::UnsignedWrap) {
    return matrix_plan_detail::Reject(desc,
                                      "compute_matrix_arithmetic_unsupported");
  }
  if (desc.element_bytes != 4u && desc.element_bytes != 8u) {
    return matrix_plan_detail::Reject(desc,
                                      "compute_matrix_element_unsupported");
  }
  if (desc.arithmetic == MatrixArithmetic::Fixed) {
    if (!ComputeFixedFormatValid(ComputeScalarForBytes(desc.element_bytes),
                                 desc.fixed_format)) {
      return matrix_plan_detail::Reject(
          desc, "compute_matrix_numeric_policy_unsupported");
    }
  } else if (!ComputeFixedFormatAbsent(desc.fixed_format)) {
    return matrix_plan_detail::Reject(
        desc, "compute_matrix_numeric_policy_unsupported");
  }
  if (desc.rows == 0u || desc.cols == 0u || desc.batch_count == 0u) {
    return matrix_plan_detail::Reject(desc, "compute_matrix_shape_zero");
  }
  if ((desc.op == MatrixOp::Mul || desc.op == MatrixOp::BatchMul) &&
      desc.inner == 0u) {
    return matrix_plan_detail::Reject(desc, "compute_matrix_shape_zero");
  }

  u64 left_count = 0u;
  u64 right_count = 0u;
  u64 output_count = 0u;
  const u64 batch = desc.op == MatrixOp::Mul ? 1u : desc.batch_count;
  if (desc.op == MatrixOp::Transpose) {
    if (!checked::mul(desc.rows, desc.cols, desc.batch_count, left_count)) {
      return matrix_plan_detail::Reject(desc, "compute_matrix_shape_overflow");
    }
    right_count = 0u;
    output_count = left_count;
  } else {
    if (!checked::mul(desc.rows, desc.inner, batch, left_count) ||
        !checked::mul(desc.inner, desc.cols, batch, right_count) ||
        !checked::mul(desc.rows, desc.cols, batch, output_count)) {
      return matrix_plan_detail::Reject(desc, "compute_matrix_shape_overflow");
    }
  }

  return MatrixPlan{
      .op = desc.op,
      .layout = desc.layout,
      .arithmetic = desc.arithmetic,
      .rows = desc.rows,
      .cols = desc.cols,
      .inner = desc.inner,
      .batch_count = batch,
      .left_count = left_count,
      .right_count = right_count,
      .output_count = output_count,
      .element_bytes = desc.element_bytes,
      .fixed_format = desc.fixed_format,
      .pass_count = 1u,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] constexpr bool
MatrixPlanMatchesDesc(const MatrixDesc &desc, const MatrixPlan &plan) noexcept {
  const u64 expected_batch = desc.op == MatrixOp::Mul ? 1u : desc.batch_count;
  return plan.ok && desc.op == plan.op && desc.layout == plan.layout &&
         desc.arithmetic == plan.arithmetic && desc.rows == plan.rows &&
         desc.cols == plan.cols && desc.inner == plan.inner &&
         expected_batch == plan.batch_count &&
         desc.element_bytes == plan.element_bytes &&
         desc.fixed_format == plan.fixed_format && plan.pass_count == 1u;
}

} // namespace rund::kernel
