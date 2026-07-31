#pragma once

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/spectrum/model.hpp>

namespace rund::kernel {
namespace spectrum_plan_detail {

constexpr u64 kMaxDimension = 16u;

[[nodiscard]] constexpr SpectrumPlan Reject(const SpectrumDesc &desc,
                                            const char *const reason) noexcept {
  return SpectrumPlan{
      .op = desc.op,
      .domain = desc.domain,
      .vectors = desc.vectors,
      .layout = desc.layout,
      .rows = desc.rows,
      .cols = desc.cols,
      .batch_count = desc.batch_count,
      .max_iterations = desc.max_iterations,
      .element_bytes = desc.element_bytes,
      .fixed_format = desc.fixed_format,
      .reason = reason,
  };
}

[[nodiscard]] constexpr u64 Min(const u64 a, const u64 b) noexcept {
  return a < b ? a : b;
}

} // namespace spectrum_plan_detail

[[nodiscard]] constexpr SpectrumPlan
PlanSpectrum(const SpectrumDesc &desc) noexcept {
  if (desc.op != SpectrumOp::SVD && desc.op != SpectrumOp::Eigen) {
    return spectrum_plan_detail::Reject(desc,
                                        "compute_spectrum_op_unsupported");
  }
  if (desc.domain != SpectrumDomain::SymmetricReal &&
      desc.domain != SpectrumDomain::GeneralReal) {
    return spectrum_plan_detail::Reject(desc,
                                        "compute_spectrum_domain_unsupported");
  }
  if (desc.vectors != SpectrumVectors::None &&
      desc.vectors != SpectrumVectors::ValuesOnly &&
      desc.vectors != SpectrumVectors::Thin &&
      desc.vectors != SpectrumVectors::Full) {
    return spectrum_plan_detail::Reject(desc,
                                        "compute_spectrum_vectors_unsupported");
  }
  if (desc.layout != MatrixLayout::RowMajor &&
      desc.layout != MatrixLayout::ColumnMajor) {
    return spectrum_plan_detail::Reject(desc,
                                        "compute_spectrum_layout_unsupported");
  }
  if (desc.element_bytes != 4u && desc.element_bytes != 8u) {
    return spectrum_plan_detail::Reject(desc,
                                        "compute_spectrum_element_unsupported");
  }
  if (!ComputePrimitiveFixedFormatValid(desc.element_bytes, desc.fixed_format,
                                        ComputeApproximation::Deterministic)) {
    return spectrum_plan_detail::Reject(
        desc, "compute_spectrum_numeric_policy_unsupported");
  }
  if (desc.rows == 0u || desc.cols == 0u || desc.batch_count == 0u) {
    return spectrum_plan_detail::Reject(desc, "compute_spectrum_shape_zero");
  }
  if (desc.rows > spectrum_plan_detail::kMaxDimension ||
      desc.cols > spectrum_plan_detail::kMaxDimension) {
    return spectrum_plan_detail::Reject(desc,
                                        "compute_spectrum_shape_dimension");
  }
  if (desc.max_iterations == 0u) {
    return spectrum_plan_detail::Reject(desc,
                                        "compute_spectrum_iterations_zero");
  }
  if (desc.op == SpectrumOp::Eigen &&
      (desc.domain != SpectrumDomain::SymmetricReal ||
       desc.rows != desc.cols)) {
    return spectrum_plan_detail::Reject(desc,
                                        "compute_spectrum_shape_symmetric");
  }

  u64 input_count = 0u;
  if (!checked::mul(desc.rows, desc.cols, desc.batch_count, input_count)) {
    return spectrum_plan_detail::Reject(desc,
                                        "compute_spectrum_shape_overflow");
  }
  const u64 value_width = desc.op == SpectrumOp::SVD
                              ? spectrum_plan_detail::Min(desc.rows, desc.cols)
                              : desc.rows;
  u64 value_count = 0u;
  if (!checked::mul(value_width, desc.batch_count)) {
    return spectrum_plan_detail::Reject(desc,
                                        "compute_spectrum_shape_overflow");
  }
  value_count = value_width * desc.batch_count;

  u64 vector_count = 0u;
  if (desc.vectors == SpectrumVectors::Thin) {
    if (!checked::mul(desc.rows, value_width, desc.batch_count, vector_count)) {
      return spectrum_plan_detail::Reject(desc,
                                          "compute_spectrum_shape_overflow");
    }
  } else if (desc.vectors == SpectrumVectors::Full) {
    if (!checked::mul(desc.rows, desc.rows, desc.batch_count, vector_count)) {
      return spectrum_plan_detail::Reject(desc,
                                          "compute_spectrum_shape_overflow");
    }
  }
  if (!checked::mul(input_count, desc.element_bytes)) {
    return spectrum_plan_detail::Reject(desc,
                                        "compute_spectrum_shape_overflow");
  }

  return SpectrumPlan{
      .op = desc.op,
      .domain = desc.domain,
      .vectors = desc.vectors,
      .layout = desc.layout,
      .rows = desc.rows,
      .cols = desc.cols,
      .batch_count = desc.batch_count,
      .input_count = input_count,
      .value_count = value_count,
      .vector_count = vector_count,
      .status_count = desc.batch_count,
      .workspace_bytes = input_count * desc.element_bytes,
      .max_iterations = desc.max_iterations,
      .element_bytes = desc.element_bytes,
      .fixed_format = desc.fixed_format,
      .pass_count = 1u,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] constexpr bool
SpectrumPlanMatchesDesc(const SpectrumDesc &desc,
                        const SpectrumPlan &plan) noexcept {
  return plan.ok && desc.op == plan.op && desc.domain == plan.domain &&
         desc.vectors == plan.vectors && desc.layout == plan.layout &&
         desc.rows == plan.rows && desc.cols == plan.cols &&
         desc.batch_count == plan.batch_count &&
         desc.max_iterations == plan.max_iterations &&
         desc.element_bytes == plan.element_bytes &&
         desc.fixed_format == plan.fixed_format && plan.pass_count == 1u;
}

} // namespace rund::kernel
