#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
dot(const ComputeValue ax, const ComputeValue ay, const ComputeValue bx,
    const ComputeValue by) noexcept {
  return add_sat(mul_fixed(ax, bx), mul_fixed(ay, by));
}

[[nodiscard]] inline ComputeValue
dot(const ComputeValue ax, const ComputeValue ay, const ComputeValue az,
    const ComputeValue bx, const ComputeValue by,
    const ComputeValue bz) noexcept {
  return add_sat(dot(ax, ay, bx, by), mul_fixed(az, bz));
}

[[nodiscard]] inline ComputeValue
dot(const ComputeValue a0, const ComputeValue a1, const ComputeValue a2,
    const ComputeValue a3, const ComputeValue b0, const ComputeValue b1,
    const ComputeValue b2, const ComputeValue b3) noexcept {
  return add_sat(dot(a0, a1, b0, b1), dot(a2, a3, b2, b3));
}

[[nodiscard]] inline ComputeValue
dot(const ComputeValue a0, const ComputeValue a1, const ComputeValue a2,
    const ComputeValue a3, const ComputeValue a4, const ComputeValue b0,
    const ComputeValue b1, const ComputeValue b2, const ComputeValue b3,
    const ComputeValue b4) noexcept {
  return add_sat(dot(a0, a1, a2, a3, b0, b1, b2, b3), mul_fixed(a4, b4));
}

[[nodiscard]] inline ComputeValue
dot(const ComputeValue a0, const ComputeValue a1, const ComputeValue a2,
    const ComputeValue a3, const ComputeValue a4, const ComputeValue a5,
    const ComputeValue b0, const ComputeValue b1, const ComputeValue b2,
    const ComputeValue b3, const ComputeValue b4,
    const ComputeValue b5) noexcept {
  return add_sat(dot(a0, a1, a2, b0, b1, b2),
                 dot(a3, a4, a5, b3, b4, b5));
}

[[nodiscard]] inline ComputeValue
dot(const ComputeValue a0, const ComputeValue a1, const ComputeValue a2,
    const ComputeValue a3, const ComputeValue a4, const ComputeValue a5,
    const ComputeValue a6, const ComputeValue a7, const ComputeValue b0,
    const ComputeValue b1, const ComputeValue b2, const ComputeValue b3,
    const ComputeValue b4, const ComputeValue b5, const ComputeValue b6,
    const ComputeValue b7) noexcept {
  return add_sat(dot(a0, a1, a2, a3, b0, b1, b2, b3),
                 dot(a4, a5, a6, a7, b4, b5, b6, b7));
}

} // namespace rund::compute_dsl
