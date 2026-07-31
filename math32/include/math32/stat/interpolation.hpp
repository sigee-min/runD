#pragma once

#include <math32/stat/rms.hpp>

#include <bit>

namespace rund::math32 {
[[nodiscard]] inline simd::I32x MulUnit(const simd::I32x lhs, const simd::I32x rhs) noexcept {
  const simd::I32x a = Clamp(lhs, simd::SplatI32(0), simd::SplatI32(FixedMax));
  const simd::I32x b = Clamp(rhs, simd::SplatI32(0), simd::SplatI32(FixedMax));
  const detail::U64x4 product =
      detail::Widen64(std::bit_cast<simd::U32x>(a)) *
      detail::Widen64(std::bit_cast<simd::U32x>(b));
  const detail::U64x4 quotient = detail::UnsignedDiv64(product, detail::SplatU64x4(static_cast<u64>(FixedMax)));
  return std::bit_cast<simd::I32x>(detail::Narrow32(quotient));
}
[[nodiscard]] inline simd::I32x Lerp(const simd::I32x lhs, const simd::I32x rhs, const simd::I32x amount) noexcept {
  const simd::I32x t = Clamp(amount, simd::SplatI32(0), simd::SplatI32(FixedMax));
  return AddSat(MulFixed(lhs, SubSat(simd::SplatI32(FixedMax), t)), MulFixed(rhs, t));
}
[[nodiscard]] inline simd::I32x ClampLerp(const simd::I32x lhs, const simd::I32x rhs, const simd::I32x amount) noexcept {
  return Lerp(lhs, rhs, Clamp(amount, simd::SplatI32(0), simd::SplatI32(FixedMax)));
}
[[nodiscard]] inline simd::I32x SmoothStep(const simd::I32x amount) noexcept {
  const simd::I32x t = Clamp(amount, simd::SplatI32(0), simd::SplatI32(FixedMax));
  const simd::I32x square = MulUnit(t, t);
  const simd::I32x cube = MulUnit(square, t);
  return SubSat(AddSat(square, AddSat(square, square)), AddSat(cube, cube));
}
[[nodiscard]] inline simd::I32x Hermite(const simd::I32x p0,
                                        const simd::I32x m0,
                                        const simd::I32x p1,
                                        const simd::I32x m1,
                                        const simd::I32x amount) noexcept {
  const simd::I32x t = Clamp(amount, simd::SplatI32(0), simd::SplatI32(FixedMax));
  const simd::I32x t2 = MulUnit(t, t);
  const simd::I32x t3 = MulUnit(t2, t);
  const simd::I32x two_t3 = AddSat(t3, t3);
  const simd::I32x three_t2 = AddSat(t2, AddSat(t2, t2));
  const simd::I32x h00 = AddSat(SubSat(two_t3, three_t2), simd::SplatI32(FixedMax));
  const simd::I32x h10 = AddSat(SubSat(t3, AddSat(t2, t2)), t);
  const simd::I32x h01 = SubSat(three_t2, two_t3);
  const simd::I32x h11 = SubSat(t3, t2);
  return AddSat(AddSat(MulFixed(p0, h00), MulFixed(m0, h10)), AddSat(MulFixed(p1, h01), MulFixed(m1, h11)));
}
}  // namespace rund::math32
