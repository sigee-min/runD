#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
standardized(const StandardizedOpQuartic, const ComputeValue value,
             const ComputeValue center, const ComputeValue scale) noexcept {
  const ComputeValue z = zscore(value, center, scale);
  const ComputeValue square = mul_fixed(z, z);
  return mul_fixed(square, square);
}

} // namespace rund::compute_dsl
