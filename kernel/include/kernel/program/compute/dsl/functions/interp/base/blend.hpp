#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue lerp(const ComputeValue lhs,
                                       const ComputeValue rhs,
                                       const ComputeValue amount) noexcept {
  const ComputeValue t = saturate(amount);
  return add_sat(mul_fixed(lhs, sub_sat(fixed_one(amount), t)),
                 mul_fixed(rhs, t));
}

[[nodiscard]] inline ComputeValue
lerp(const ComputeValue x00, const ComputeValue x10, const ComputeValue x01,
     const ComputeValue x11, const ComputeValue tx,
     const ComputeValue ty) noexcept {
  return lerp(lerp(x00, x10, tx), lerp(x01, x11, tx), ty);
}

[[nodiscard]] inline ComputeValue
lerp(const ComputeValue x000, const ComputeValue x100,
     const ComputeValue x010, const ComputeValue x110,
     const ComputeValue x001, const ComputeValue x101,
     const ComputeValue x011, const ComputeValue x111, const ComputeValue tx,
     const ComputeValue ty, const ComputeValue tz) noexcept {
  return lerp(lerp(x000, x100, x010, x110, tx, ty),
              lerp(x001, x101, x011, x111, tx, ty), tz);
}

} // namespace rund::compute_dsl
