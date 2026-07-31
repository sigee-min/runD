#pragma once

#include <kernel/program/compute/dsl/axis.hpp>
#include <kernel/program/compute/dsl/functions/geometry/barycentric/weight.hpp>
#include <kernel/program/compute/dsl/functions/geometry/orientation.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
bary(Axis::XTag, const ComputeValue px, const ComputeValue py,
     const ComputeValue ax, const ComputeValue ay, const ComputeValue bx,
     const ComputeValue by, const ComputeValue cx,
     const ComputeValue cy) noexcept {
  const ComputeValue denom = orient(ax, ay, bx, by, cx, cy);
  return detail::BarycentricWeight(orient(px, py, bx, by, cx, cy), denom);
}

[[nodiscard]] inline ComputeValue
bary(Axis::YTag, const ComputeValue px, const ComputeValue py,
     const ComputeValue ax, const ComputeValue ay, const ComputeValue bx,
     const ComputeValue by, const ComputeValue cx,
     const ComputeValue cy) noexcept {
  const ComputeValue denom = orient(ax, ay, bx, by, cx, cy);
  return detail::BarycentricWeight(orient(ax, ay, px, py, cx, cy), denom);
}

[[nodiscard]] inline ComputeValue
bary(Axis::ZTag, const ComputeValue px, const ComputeValue py,
     const ComputeValue ax, const ComputeValue ay, const ComputeValue bx,
     const ComputeValue by, const ComputeValue cx,
     const ComputeValue cy) noexcept {
  const ComputeValue denom = orient(ax, ay, bx, by, cx, cy);
  return detail::BarycentricWeight(orient(ax, ay, bx, by, px, py), denom);
}

} // namespace rund::compute_dsl
