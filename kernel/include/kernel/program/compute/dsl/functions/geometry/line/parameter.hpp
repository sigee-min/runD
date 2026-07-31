#pragma once

#include <kernel/program/compute/dsl/geometry.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
line(const GeometryOpParameter, const ComputeValue px, const ComputeValue py,
     const ComputeValue ax, const ComputeValue ay, const ComputeValue bx,
     const ComputeValue by) noexcept {
  const ComputeValue dx = sub_sat(bx, ax);
  const ComputeValue dy = sub_sat(by, ay);
  const ComputeValue qx = sub_sat(px, ax);
  const ComputeValue qy = sub_sat(py, ay);
  return detail::GeometryRatio(dot(qx, qy, dx, dy),
                               len(MetricOp::Squared, dx, dy));
}

} // namespace rund::compute_dsl
