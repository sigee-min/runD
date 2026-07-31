#pragma once

#include <kernel/program/compute/dsl/axis.hpp>
#include <kernel/program/compute/dsl/functions/geometry/line/parameter.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
line(const GeometryOpProjection, Axis::XTag, const ComputeValue px,
     const ComputeValue py, const ComputeValue ax, const ComputeValue ay,
     const ComputeValue bx, const ComputeValue by) noexcept {
  const ComputeValue t = line(LineOp::Parameter, px, py, ax, ay, bx, by);
  return add_sat(ax, mul_fixed(sub_sat(bx, ax), t));
}

[[nodiscard]] inline ComputeValue
line(const GeometryOpProjection, Axis::YTag, const ComputeValue px,
     const ComputeValue py, const ComputeValue ax, const ComputeValue ay,
     const ComputeValue bx, const ComputeValue by) noexcept {
  const ComputeValue t = line(LineOp::Parameter, px, py, ax, ay, bx, by);
  return add_sat(ay, mul_fixed(sub_sat(by, ay), t));
}

} // namespace rund::compute_dsl
