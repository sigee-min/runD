#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
mean(const CenteredOpCubic op, const ComputeValue lhs,
     const ComputeValue rhs) noexcept {
  const ComputeValue center = mean(lhs, rhs);
  return mean(centered(op, lhs, center), centered(op, rhs, center));
}

[[nodiscard]] inline ComputeValue
mean(const CenteredOpCubic op, const ComputeValue a, const ComputeValue b,
     const ComputeValue c) noexcept {
  const ComputeValue center = mean(a, b, c);
  return mean(centered(op, a, center), centered(op, b, center),
              centered(op, c, center));
}

[[nodiscard]] inline ComputeValue
mean(const CenteredOpCubic op, const ComputeValue a, const ComputeValue b,
     const ComputeValue c, const ComputeValue d) noexcept {
  const ComputeValue center = mean(a, b, c, d);
  return mean(centered(op, a, center), centered(op, b, center),
              centered(op, c, center), centered(op, d, center));
}

} // namespace rund::compute_dsl
