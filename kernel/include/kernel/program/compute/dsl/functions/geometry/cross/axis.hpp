#pragma once

#include <kernel/program/compute/dsl/axis.hpp>
#include <kernel/program/compute/dsl/functions/geometry/cross/scalar.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
cross(Axis::XTag, const ComputeValue ax, const ComputeValue ay,
      const ComputeValue az, const ComputeValue bx, const ComputeValue by,
      const ComputeValue bz) noexcept {
  (void)ax;
  (void)bx;
  return cross(ay, az, by, bz);
}

[[nodiscard]] inline ComputeValue
cross(Axis::YTag, const ComputeValue ax, const ComputeValue ay,
      const ComputeValue az, const ComputeValue bx, const ComputeValue by,
      const ComputeValue bz) noexcept {
  (void)ay;
  (void)by;
  return cross(az, ax, bz, bx);
}

[[nodiscard]] inline ComputeValue
cross(Axis::ZTag, const ComputeValue ax, const ComputeValue ay,
      const ComputeValue az, const ComputeValue bx, const ComputeValue by,
      const ComputeValue bz) noexcept {
  (void)az;
  (void)bz;
  return cross(ax, ay, bx, by);
}

} // namespace rund::compute_dsl
