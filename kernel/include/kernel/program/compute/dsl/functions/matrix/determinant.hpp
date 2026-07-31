#pragma once

#include <kernel/program/compute/dsl/matrix.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
mat(const MatOpDeterminant, const ComputeValue m00, const ComputeValue m01,
    const ComputeValue m10, const ComputeValue m11) noexcept {
  return sub_sat(mul_fixed(m00, m11), mul_fixed(m01, m10));
}

[[nodiscard]] inline ComputeValue
mat(const MatOpDeterminant, const ComputeValue m00, const ComputeValue m01,
    const ComputeValue m02, const ComputeValue m10, const ComputeValue m11,
    const ComputeValue m12, const ComputeValue m20, const ComputeValue m21,
    const ComputeValue m22) noexcept {
  const ComputeValue a =
      mul_fixed(m00, mat(MatOp::Determinant, m11, m12, m21, m22));
  const ComputeValue b =
      mul_fixed(m01, mat(MatOp::Determinant, m10, m12, m20, m22));
  const ComputeValue c =
      mul_fixed(m02, mat(MatOp::Determinant, m10, m11, m20, m21));
  return add_sat(sub_sat(a, b), c);
}

} // namespace rund::compute_dsl
