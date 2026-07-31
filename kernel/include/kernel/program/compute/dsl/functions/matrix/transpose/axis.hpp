#pragma once

#include <kernel/program/compute/dsl/axis.hpp>
#include <kernel/program/compute/dsl/matrix.hpp>
#include <kernel/program/compute/dsl/functions/linear/dot/value.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
mat(const MatOpTranspose, Axis::XTag, const ComputeValue m00,
    const ComputeValue m10, const ComputeValue x,
    const ComputeValue y) noexcept {
  return dot(m00, m10, x, y);
}

[[nodiscard]] inline ComputeValue
mat(const MatOpTranspose, Axis::YTag, const ComputeValue m01,
    const ComputeValue m11, const ComputeValue x,
    const ComputeValue y) noexcept {
  return dot(m01, m11, x, y);
}

[[nodiscard]] inline ComputeValue
mat(const MatOpTranspose, Axis::XTag, const ComputeValue m00,
    const ComputeValue m10, const ComputeValue m20, const ComputeValue x,
    const ComputeValue y, const ComputeValue z) noexcept {
  return dot(m00, m10, m20, x, y, z);
}

[[nodiscard]] inline ComputeValue
mat(const MatOpTranspose, Axis::YTag, const ComputeValue m01,
    const ComputeValue m11, const ComputeValue m21, const ComputeValue x,
    const ComputeValue y, const ComputeValue z) noexcept {
  return dot(m01, m11, m21, x, y, z);
}

[[nodiscard]] inline ComputeValue
mat(const MatOpTranspose, Axis::ZTag, const ComputeValue m02,
    const ComputeValue m12, const ComputeValue m22, const ComputeValue x,
    const ComputeValue y, const ComputeValue z) noexcept {
  return dot(m02, m12, m22, x, y, z);
}

} // namespace rund::compute_dsl
