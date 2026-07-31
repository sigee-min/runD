#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
triple(const ComputeValue ax, const ComputeValue ay, const ComputeValue az,
       const ComputeValue bx, const ComputeValue by, const ComputeValue bz,
       const ComputeValue cx, const ComputeValue cy,
       const ComputeValue cz) noexcept {
  return dot(ax, ay, az, cross(Axis::X, bx, by, bz, cx, cy, cz),
             cross(Axis::Y, bx, by, bz, cx, cy, cz),
             cross(Axis::Z, bx, by, bz, cx, cy, cz));
}

} // namespace rund::compute_dsl
