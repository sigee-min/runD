#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
poly_deriv(const ComputeValue x, const ComputeValue c1,
           const ComputeValue c2) noexcept {
  return add_sat(c1, mul_fixed(x, add_sat(c2, c2)));
}

[[nodiscard]] inline ComputeValue
poly_deriv(const ComputeValue x, const ComputeValue c1, const ComputeValue c2,
           const ComputeValue c3) noexcept {
  const ComputeValue two_c2 = add_sat(c2, c2);
  const ComputeValue three_c3 = add_sat(add_sat(c3, c3), c3);
  return add_sat(c1, mul_fixed(x, add_sat(two_c2, mul_fixed(three_c3, x))));
}

} // namespace rund::compute_dsl
