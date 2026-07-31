#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue var(const ComputeValue lhs,
                                      const ComputeValue rhs) noexcept {
  const ComputeValue mean_value = mean(lhs, rhs);
  return mean(centered(CenteredOp::Squared, lhs, mean_value),
              centered(CenteredOp::Squared, rhs, mean_value));
}

[[nodiscard]] inline ComputeValue var(const ComputeValue a,
                                      const ComputeValue b,
                                      const ComputeValue c) noexcept {
  const ComputeValue mean_value = mean(a, b, c);
  return mean(centered(CenteredOp::Squared, a, mean_value),
              centered(CenteredOp::Squared, b, mean_value),
              centered(CenteredOp::Squared, c, mean_value));
}

[[nodiscard]] inline ComputeValue
var(const ComputeValue a, const ComputeValue b, const ComputeValue c,
    const ComputeValue d) noexcept {
  const ComputeValue mean_value = mean(a, b, c, d);
  return mean(centered(CenteredOp::Squared, a, mean_value),
              centered(CenteredOp::Squared, b, mean_value),
              centered(CenteredOp::Squared, c, mean_value),
              centered(CenteredOp::Squared, d, mean_value));
}

} // namespace rund::compute_dsl
