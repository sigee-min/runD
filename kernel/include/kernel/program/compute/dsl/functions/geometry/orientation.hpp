#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
orient(const ComputeValue ax, const ComputeValue ay, const ComputeValue bx,
       const ComputeValue by, const ComputeValue cx,
       const ComputeValue cy) noexcept {
  return cross(sub_sat(bx, ax), sub_sat(by, ay), sub_sat(cx, ax),
               sub_sat(cy, ay));
}

} // namespace rund::compute_dsl
