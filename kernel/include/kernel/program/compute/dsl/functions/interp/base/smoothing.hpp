#pragma once

namespace rund::compute_dsl {

namespace detail {

[[nodiscard]] inline ComputeValue SmootherScale5(
    const ComputeValue value) noexcept {
  const ComputeValue two = add_sat(value, value);
  const ComputeValue four = add_sat(two, two);
  return add_sat(four, value);
}

[[nodiscard]] inline ComputeValue SmootherScale10(
    const ComputeValue value) noexcept {
  const ComputeValue five = SmootherScale5(value);
  return add_sat(five, five);
}

} // namespace detail

[[nodiscard]] inline ComputeValue fade(const ComputeValue amount) noexcept {
  const ComputeValue t = saturate(amount);
  const ComputeValue t2 = mul_fixed(t, t);
  const ComputeValue inv = sub_sat(fixed_one(t), t);
  const ComputeValue bump = mul_fixed(t2, inv);
  return add_sat(t2, add_sat(bump, bump));
}

[[nodiscard]] inline ComputeValue
smoothstep(const ComputeValue edge0, const ComputeValue edge1,
           const ComputeValue value) noexcept {
  return fade(unlerp(edge0, edge1, value));
}

[[nodiscard]] inline ComputeValue
smootherstep(const ComputeValue edge0, const ComputeValue edge1,
             const ComputeValue value) noexcept {
  const ComputeValue t = unlerp(edge0, edge1, value);
  const ComputeValue inv = sub_sat(fixed_one(t), t);
  const ComputeValue t2 = mul_fixed(t, t);
  const ComputeValue t3 = mul_fixed(t2, t);
  const ComputeValue t4 = mul_fixed(t3, t);
  const ComputeValue t5 = mul_fixed(t4, t);
  const ComputeValue inv2 = mul_fixed(inv, inv);
  const ComputeValue first = mul_fixed(t3, inv2);
  const ComputeValue second = mul_fixed(t4, inv);
  return add_sat(add_sat(detail::SmootherScale10(first),
                         detail::SmootherScale5(second)),
                 t5);
}

} // namespace rund::compute_dsl
