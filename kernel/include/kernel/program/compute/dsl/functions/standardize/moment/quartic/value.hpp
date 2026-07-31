#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
mean(const StandardizedOpQuartic op, const ComputeValue lhs,
     const ComputeValue rhs) noexcept {
  const ComputeValue center = mean(lhs, rhs);
  const ComputeValue scale = detail::standard_scale(lhs, rhs);
  return mean(standardized(op, lhs, center, scale),
              standardized(op, rhs, center, scale));
}

[[nodiscard]] inline ComputeValue
mean(const StandardizedOpQuartic op, const ComputeValue a, const ComputeValue b,
     const ComputeValue c) noexcept {
  const ComputeValue center = mean(a, b, c);
  const ComputeValue scale = detail::standard_scale(a, b, c);
  return mean(standardized(op, a, center, scale),
              standardized(op, b, center, scale),
              standardized(op, c, center, scale));
}

[[nodiscard]] inline ComputeValue
mean(const StandardizedOpQuartic op, const ComputeValue a, const ComputeValue b,
     const ComputeValue c, const ComputeValue d) noexcept {
  const ComputeValue center = mean(a, b, c, d);
  const ComputeValue scale = detail::standard_scale(a, b, c, d);
  return mean(standardized(op, a, center, scale),
              standardized(op, b, center, scale),
              standardized(op, c, center, scale),
              standardized(op, d, center, scale));
}

} // namespace rund::compute_dsl
