#pragma once

#include <math64/fixed/lane.hpp>

namespace rund::math64 {
namespace detail {

[[nodiscard]] inline U128x2 UnsignedSqrt128(U128x2 value) noexcept {
  U128x2 result = SplatU128x2(0);
  U128x2 bit = SplatU128x2(u128{1} << 126u);
  for (int step = 0; step < 64; ++step) {
    const Mask128x2 ge = Ge128(value, result + bit);
    value = Select128(ge, value - (result + bit), value);
    result = Select128(ge, (result >> 1u) + bit, result >> 1u);
    bit >>= 2u;
  }
  return result;
}
}  // namespace detail

[[nodiscard]] inline simd::I64x Sqrt(const simd::I64x value) noexcept {
  const detail::Mask128x2 positive =
      detail::Gt128(detail::Widen128(value), detail::SplatI128x2(0));
  const detail::U128x2 magnitude =
      detail::Widen128(std::bit_cast<simd::U64x>(value));
  const detail::U128x2 radicand = detail::Select128(positive, magnitude, detail::SplatU128x2(0)) << 63u;
  return std::bit_cast<simd::I64x>(
      detail::Narrow64(detail::UnsignedSqrt128(radicand)));
}
[[nodiscard]] inline simd::I64x Rsqrt(const simd::I64x value) noexcept {
  const simd::I64x root = Sqrt(value);
  return simd::Select(simd::Le(value, simd::SplatI64(0)), simd::SplatI64(FixedMax), Recip(root));
}

}  // namespace rund::math64
