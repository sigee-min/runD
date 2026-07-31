#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
unlerp(const ComputeValue lo, const ComputeValue hi,
       const ComputeValue value) noexcept {
  const ComputeValue span = sub_sat(hi, lo);
  const ComputeValue offset = sub_sat(value, lo);
  return select(le(hi, lo), fixed_zero(value),
                saturate(div_fixed(offset, span)));
}

[[nodiscard]] inline ComputeValue
remap(const ComputeValue in_lo, const ComputeValue in_hi,
      const ComputeValue out_lo, const ComputeValue out_hi,
      const ComputeValue value) noexcept {
  return lerp(out_lo, out_hi, unlerp(in_lo, in_hi, value));
}

} // namespace rund::compute_dsl
