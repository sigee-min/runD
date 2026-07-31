#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue mean(const MeanOpSquared,
                                       const ComputeValue lhs,
                                       const ComputeValue rhs) noexcept {
  return mean(mul_fixed(lhs, lhs), mul_fixed(rhs, rhs));
}

[[nodiscard]] inline ComputeValue mean(const MeanOpSquared,
                                       const ComputeValue a,
                                       const ComputeValue b,
                                       const ComputeValue c) noexcept {
  return mean(mul_fixed(a, a), mul_fixed(b, b), mul_fixed(c, c));
}

[[nodiscard]] inline ComputeValue
mean(const MeanOpSquared, const ComputeValue a, const ComputeValue b,
     const ComputeValue c, const ComputeValue d) noexcept {
  return mean(mul_fixed(a, a), mul_fixed(b, b), mul_fixed(c, c),
              mul_fixed(d, d));
}

} // namespace rund::compute_dsl
