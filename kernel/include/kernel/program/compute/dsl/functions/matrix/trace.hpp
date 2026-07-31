#pragma once

#include <kernel/program/compute/dsl/matrix.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue mat(const MatOpTrace, const ComputeValue m00,
                                      const ComputeValue m11) noexcept {
  return add_sat(m00, m11);
}

[[nodiscard]] inline ComputeValue
mat(const MatOpTrace, const ComputeValue m00, const ComputeValue m11,
    const ComputeValue m22) noexcept {
  return add_sat(add_sat(m00, m11), m22);
}

} // namespace rund::compute_dsl
