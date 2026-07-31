#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
centered(const CenteredOpCubic, const ComputeValue value,
         const ComputeValue center) noexcept {
  const ComputeValue delta = centered(value, center);
  return mul_fixed(mul_fixed(delta, delta), delta);
}

} // namespace rund::compute_dsl
