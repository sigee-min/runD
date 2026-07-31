#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue mean(const ComputeValue lhs,
                                        const ComputeValue rhs) noexcept {
  const ComputeValue half = fixed(FixedOp::Half, lhs);
  return add_sat(mul_fixed(lhs, half), mul_fixed(rhs, half));
}

[[nodiscard]] inline ComputeValue mean(const ComputeValue a,
                                        const ComputeValue b,
                                        const ComputeValue c) noexcept {
  const ComputeValue third = fixed(FixedOp::Third, a);
  return add_sat(add_sat(mul_fixed(a, third), mul_fixed(b, third)),
                 mul_fixed(c, third));
}

[[nodiscard]] inline ComputeValue
mean(const ComputeValue a, const ComputeValue b, const ComputeValue c,
     const ComputeValue d) noexcept {
  const ComputeValue quarter = fixed(FixedOp::Quarter, a);
  return add_sat(add_sat(mul_fixed(a, quarter), mul_fixed(b, quarter)),
                 add_sat(mul_fixed(c, quarter), mul_fixed(d, quarter)));
}

} // namespace rund::compute_dsl
