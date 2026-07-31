#pragma once

#include <kernel/program/compute/dsl/geometry.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
plane(const GeometryOpParameter, const ComputeValue px, const ComputeValue py,
      const ComputeValue pz, const ComputeValue ax, const ComputeValue ay,
      const ComputeValue az, const ComputeValue nx, const ComputeValue ny,
      const ComputeValue nz) noexcept {
  const ComputeValue qx = sub_sat(px, ax);
  const ComputeValue qy = sub_sat(py, ay);
  const ComputeValue qz = sub_sat(pz, az);
  return detail::GeometryRatio(dot(qx, qy, qz, nx, ny, nz),
                               len(MetricOp::Squared, nx, ny, nz));
}

} // namespace rund::compute_dsl
