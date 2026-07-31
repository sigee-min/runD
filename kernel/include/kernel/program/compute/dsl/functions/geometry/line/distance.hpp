#pragma once

#include <kernel/program/compute/dsl/geometry.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
line(const GeometryOpDistance, const MetricOpSquared, const ComputeValue px,
     const ComputeValue py, const ComputeValue ax, const ComputeValue ay,
     const ComputeValue bx, const ComputeValue by) noexcept {
  const ComputeValue dx = sub_sat(bx, ax);
  const ComputeValue dy = sub_sat(by, ay);
  const ComputeValue qx = sub_sat(px, ax);
  const ComputeValue qy = sub_sat(py, ay);
  const ComputeValue area = cross(dx, dy, qx, qy);
  return detail::GeometryRatio(mul_fixed(area, area),
                               len(MetricOp::Squared, dx, dy));
}

[[nodiscard]] inline ComputeValue
line(const GeometryOpDistance, const ComputeValue px, const ComputeValue py,
     const ComputeValue ax, const ComputeValue ay, const ComputeValue bx,
     const ComputeValue by) noexcept {
  return sqrt(line(LineOp::Distance, MetricOp::Squared, px, py, ax, ay, bx,
                   by));
}

} // namespace rund::compute_dsl
