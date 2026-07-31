#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
window(const WindowOpParabolic, const ComputeValue amount) noexcept {
  const ComputeValue t = saturate(amount);
  const ComputeValue q = mul_fixed(t, sub_sat(fixed_one(t), t));
  return add_sat(add_sat(q, q), add_sat(q, q));
}

} // namespace rund::compute_dsl
