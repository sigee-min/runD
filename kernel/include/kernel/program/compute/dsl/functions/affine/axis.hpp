#pragma once

#include <kernel/program/compute/dsl/functions/matrix/rows/axis.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
aff(Axis::XTag axis, const ComputeValue m00, const ComputeValue m01,
    const ComputeValue tx, const ComputeValue x,
    const ComputeValue y) noexcept {
  return add_sat(mat(axis, m00, m01, x, y), tx);
}

[[nodiscard]] inline ComputeValue
aff(Axis::YTag axis, const ComputeValue m10, const ComputeValue m11,
    const ComputeValue ty, const ComputeValue x,
    const ComputeValue y) noexcept {
  return add_sat(mat(axis, m10, m11, x, y), ty);
}

[[nodiscard]] inline ComputeValue
aff(Axis::XTag axis, const ComputeValue m00, const ComputeValue m01,
    const ComputeValue m02, const ComputeValue tx, const ComputeValue x,
    const ComputeValue y, const ComputeValue z) noexcept {
  return add_sat(mat(axis, m00, m01, m02, x, y, z), tx);
}

[[nodiscard]] inline ComputeValue
aff(Axis::YTag axis, const ComputeValue m10, const ComputeValue m11,
    const ComputeValue m12, const ComputeValue ty, const ComputeValue x,
    const ComputeValue y, const ComputeValue z) noexcept {
  return add_sat(mat(axis, m10, m11, m12, x, y, z), ty);
}

[[nodiscard]] inline ComputeValue
aff(Axis::ZTag axis, const ComputeValue m20, const ComputeValue m21,
    const ComputeValue m22, const ComputeValue tz, const ComputeValue x,
    const ComputeValue y, const ComputeValue z) noexcept {
  return add_sat(mat(axis, m20, m21, m22, x, y, z), tz);
}

} // namespace rund::compute_dsl
