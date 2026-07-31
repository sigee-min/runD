#pragma once

#include <kernel/program/compute/dsl/axis.hpp>
#include <kernel/program/compute/dsl/matrix.hpp>
#include <kernel/program/compute/dsl/functions/matrix/determinant.hpp>
#include <kernel/program/compute/dsl/functions/matrix/solve/component.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
mat(const MatOpSolve, Axis::XTag, const ComputeValue m00,
    const ComputeValue m01, const ComputeValue m10, const ComputeValue m11,
    const ComputeValue b0, const ComputeValue b1) noexcept {
  const ComputeValue det = mat(MatOp::Determinant, m00, m01, m10, m11);
  return detail::MatrixSolveComponent(
      mat(MatOp::Determinant, b0, m01, b1, m11), det);
}

[[nodiscard]] inline ComputeValue
mat(const MatOpSolve, Axis::YTag, const ComputeValue m00,
    const ComputeValue m01, const ComputeValue m10, const ComputeValue m11,
    const ComputeValue b0, const ComputeValue b1) noexcept {
  const ComputeValue det = mat(MatOp::Determinant, m00, m01, m10, m11);
  return detail::MatrixSolveComponent(
      mat(MatOp::Determinant, m00, b0, m10, b1), det);
}

} // namespace rund::compute_dsl
