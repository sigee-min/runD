#pragma once

namespace rund::compute_dsl::detail {

[[nodiscard]] inline ComputeValue
MatrixSolveComponent(const ComputeValue numerator,
                     const ComputeValue det) noexcept {
  return select(eq(det, 0), fixed_zero(numerator), div_fixed(numerator, det));
}

} // namespace rund::compute_dsl::detail
