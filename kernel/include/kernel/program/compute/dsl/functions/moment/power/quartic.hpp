#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
centered(const CenteredOpQuartic, const ComputeValue value,
         const ComputeValue center) noexcept {
  const ComputeValue square = centered(CenteredOp::Squared, value, center);
  return mul_fixed(square, square);
}

} // namespace rund::compute_dsl
