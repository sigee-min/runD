#pragma once

#include <kernel/program/compute/dsl/functions/geometry/projection/component/axis.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
reject(Axis::XTag axis, const ComputeValue ax, const ComputeValue ay,
       const ComputeValue bx, const ComputeValue by) noexcept {
  return sub_sat(ax, proj(axis, ax, ay, bx, by));
}

[[nodiscard]] inline ComputeValue
reject(Axis::YTag axis, const ComputeValue ax, const ComputeValue ay,
       const ComputeValue bx, const ComputeValue by) noexcept {
  return sub_sat(ay, proj(axis, ax, ay, bx, by));
}

[[nodiscard]] inline ComputeValue
reject(Axis::XTag axis, const ComputeValue ax, const ComputeValue ay,
       const ComputeValue az, const ComputeValue bx, const ComputeValue by,
       const ComputeValue bz) noexcept {
  return sub_sat(ax, proj(axis, ax, ay, az, bx, by, bz));
}

[[nodiscard]] inline ComputeValue
reject(Axis::YTag axis, const ComputeValue ax, const ComputeValue ay,
       const ComputeValue az, const ComputeValue bx, const ComputeValue by,
       const ComputeValue bz) noexcept {
  return sub_sat(ay, proj(axis, ax, ay, az, bx, by, bz));
}

[[nodiscard]] inline ComputeValue
reject(Axis::ZTag axis, const ComputeValue ax, const ComputeValue ay,
       const ComputeValue az, const ComputeValue bx, const ComputeValue by,
       const ComputeValue bz) noexcept {
  return sub_sat(az, proj(axis, ax, ay, az, bx, by, bz));
}

} // namespace rund::compute_dsl
