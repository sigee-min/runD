#pragma once

#include <math32/fixed/lane.hpp>

namespace rund::math32 {
namespace detail {

[[nodiscard]] inline U64x4 UnsignedSqrtWide(U64x4 value) noexcept {
  U64x4 result = SplatU64x4(0);
  U64x4 bit = SplatU64x4(u64{1} << 62u);
  for (int step = 0; step < 32; ++step) {
    const Mask64x4 ge = Ge64(value, result + bit);
    value = Select64(ge, value - (result + bit), value);
    result = Select64(ge, (result >> 1u) + bit, result >> 1u);
    bit >>= 2u;
  }
  return result;
}
}  // namespace detail

[[nodiscard]] inline simd::I32x Sqrt(const simd::I32x value) noexcept {
  const detail::Mask64x4 positive =
      detail::Gt64(detail::Widen64(value), detail::SplatI64x4(0));
  const detail::U64x4 magnitude =
      detail::Widen64(std::bit_cast<simd::U32x>(value));
  return detail::ClampU64x4ToI32x(detail::UnsignedSqrtWide(detail::Select64(positive, magnitude, detail::SplatU64x4(0)) << 31u));
}
[[nodiscard]] inline simd::I32x Rsqrt(const simd::I32x value) noexcept {
  const simd::I32x root = Sqrt(value);
  return simd::Select(simd::Le(value, simd::SplatI32(0)), simd::SplatI32(FixedMax), Recip(root));
}

}  // namespace rund::math32
