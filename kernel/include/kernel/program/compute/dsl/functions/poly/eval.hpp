#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
poly(const ComputeValue x, const ComputeValue c0, const ComputeValue c1,
     const ComputeValue c2) noexcept {
  return add_sat(c0, mul_fixed(x, add_sat(c1, mul_fixed(c2, x))));
}

[[nodiscard]] inline ComputeValue
poly(const ComputeValue x, const ComputeValue c0, const ComputeValue c1,
     const ComputeValue c2, const ComputeValue c3) noexcept {
  return add_sat(
      c0, mul_fixed(
              x, add_sat(c1,
                         mul_fixed(x, add_sat(c2, mul_fixed(c3, x))))));
}

} // namespace rund::compute_dsl
