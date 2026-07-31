#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
standardized(const StandardizedOpCubic, const ComputeValue value,
             const ComputeValue center, const ComputeValue scale) noexcept {
  const ComputeValue z = zscore(value, center, scale);
  return mul_fixed(mul_fixed(z, z), z);
}

} // namespace rund::compute_dsl
