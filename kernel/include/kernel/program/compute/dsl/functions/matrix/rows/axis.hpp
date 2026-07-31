#pragma once

#include <kernel/program/compute/dsl/axis.hpp>
#include <kernel/program/compute/dsl/functions/linear/dot/value.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue mat(Axis::XTag, const ComputeValue m00,
                                      const ComputeValue m01,
                                      const ComputeValue x,
                                      const ComputeValue y) noexcept {
  return dot(m00, m01, x, y);
}

[[nodiscard]] inline ComputeValue mat(Axis::YTag, const ComputeValue m10,
                                      const ComputeValue m11,
                                      const ComputeValue x,
                                      const ComputeValue y) noexcept {
  return dot(m10, m11, x, y);
}

[[nodiscard]] inline ComputeValue mat(Axis::XTag, const ComputeValue m00,
                                      const ComputeValue m01,
                                      const ComputeValue m02,
                                      const ComputeValue x,
                                      const ComputeValue y,
                                      const ComputeValue z) noexcept {
  return dot(m00, m01, m02, x, y, z);
}

[[nodiscard]] inline ComputeValue mat(Axis::YTag, const ComputeValue m10,
                                      const ComputeValue m11,
                                      const ComputeValue m12,
                                      const ComputeValue x,
                                      const ComputeValue y,
                                      const ComputeValue z) noexcept {
  return dot(m10, m11, m12, x, y, z);
}

[[nodiscard]] inline ComputeValue mat(Axis::ZTag, const ComputeValue m20,
                                      const ComputeValue m21,
                                      const ComputeValue m22,
                                      const ComputeValue x,
                                      const ComputeValue y,
                                      const ComputeValue z) noexcept {
  return dot(m20, m21, m22, x, y, z);
}

} // namespace rund::compute_dsl
