#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue rms(const ComputeValue lhs,
                                      const ComputeValue rhs) noexcept {
  return sqrt(mean(MeanOp::Squared, lhs, rhs));
}

[[nodiscard]] inline ComputeValue rms(const ComputeValue a,
                                      const ComputeValue b,
                                      const ComputeValue c) noexcept {
  return sqrt(mean(MeanOp::Squared, a, b, c));
}

[[nodiscard]] inline ComputeValue
rms(const ComputeValue a, const ComputeValue b, const ComputeValue c,
    const ComputeValue d) noexcept {
  return sqrt(mean(MeanOp::Squared, a, b, c, d));
}

} // namespace rund::compute_dsl
