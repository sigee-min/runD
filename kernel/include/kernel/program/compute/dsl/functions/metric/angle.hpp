#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
angle(const AngleOpCosine, const ComputeValue ax, const ComputeValue ay,
      const ComputeValue bx, const ComputeValue by) noexcept {
  const ComputeValue denom =
      sqrt(mul_fixed(len(MetricOp::Squared, ax, ay),
                     len(MetricOp::Squared, bx, by)));
  return select(eq(denom, 0), fixed_zero(ax),
                div_fixed(dot(ax, ay, bx, by), denom));
}

[[nodiscard]] inline ComputeValue
angle(const AngleOpCosine, const ComputeValue ax, const ComputeValue ay,
      const ComputeValue az, const ComputeValue bx, const ComputeValue by,
      const ComputeValue bz) noexcept {
  const ComputeValue denom =
      sqrt(mul_fixed(len(MetricOp::Squared, ax, ay, az),
                     len(MetricOp::Squared, bx, by, bz)));
  return select(eq(denom, 0), fixed_zero(ax),
                div_fixed(dot(ax, ay, az, bx, by, bz), denom));
}

} // namespace rund::compute_dsl
