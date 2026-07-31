#pragma once

#include <kernel/program/compute/dsl/axis.hpp>
#include <kernel/program/compute/dsl/functions/geometry/plane/parameter.hpp>
#include <kernel/program/compute/dsl/functions/geometry/plane/projection/component.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
plane(const GeometryOpProjection, Axis::XTag, const ComputeValue px,
      const ComputeValue py, const ComputeValue pz, const ComputeValue ax,
      const ComputeValue ay, const ComputeValue az, const ComputeValue nx,
      const ComputeValue ny, const ComputeValue nz) noexcept {
  const ComputeValue t = plane(PlaneOp::Parameter, px, py, pz, ax, ay, az, nx,
                               ny, nz);
  return detail::PlaneProjectionComponent(px, nx, t);
}

[[nodiscard]] inline ComputeValue
plane(const GeometryOpProjection, Axis::YTag, const ComputeValue px,
      const ComputeValue py, const ComputeValue pz, const ComputeValue ax,
      const ComputeValue ay, const ComputeValue az, const ComputeValue nx,
      const ComputeValue ny, const ComputeValue nz) noexcept {
  const ComputeValue t = plane(PlaneOp::Parameter, px, py, pz, ax, ay, az, nx,
                               ny, nz);
  return detail::PlaneProjectionComponent(py, ny, t);
}

[[nodiscard]] inline ComputeValue
plane(const GeometryOpProjection, Axis::ZTag, const ComputeValue px,
      const ComputeValue py, const ComputeValue pz, const ComputeValue ax,
      const ComputeValue ay, const ComputeValue az, const ComputeValue nx,
      const ComputeValue ny, const ComputeValue nz) noexcept {
  const ComputeValue t = plane(PlaneOp::Parameter, px, py, pz, ax, ay, az, nx,
                               ny, nz);
  return detail::PlaneProjectionComponent(pz, nz, t);
}

} // namespace rund::compute_dsl
