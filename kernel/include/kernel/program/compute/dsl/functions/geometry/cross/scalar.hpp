#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
cross(const ComputeValue ax, const ComputeValue ay, const ComputeValue bx,
      const ComputeValue by) noexcept {
  return sub_sat(mul_fixed(ax, by), mul_fixed(ay, bx));
}

} // namespace rund::compute_dsl
