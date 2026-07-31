#pragma once

#include <math64/stat/rms.hpp>

#include <bit>

namespace rund::math64 {
[[nodiscard]] inline simd::I64x MulUnit(const simd::I64x lhs, const simd::I64x rhs) noexcept {
  const simd::I64x a = Clamp(lhs, simd::SplatI64(0), simd::SplatI64(FixedMax));
  const simd::I64x b = Clamp(rhs, simd::SplatI64(0), simd::SplatI64(FixedMax));
  const detail::U128x2 product =
      detail::Widen128(std::bit_cast<simd::U64x>(a)) *
      detail::Widen128(std::bit_cast<simd::U64x>(b));
  const detail::U128x2 quotient = detail::UnsignedDiv128(product, detail::SplatU128x2(static_cast<u64>(FixedMax)));
  return std::bit_cast<simd::I64x>(detail::Narrow64(quotient));
}
[[nodiscard]] inline simd::I64x Lerp(const simd::I64x lhs, const simd::I64x rhs, const simd::I64x amount) noexcept {
  const simd::I64x t = Clamp(amount, simd::SplatI64(0), simd::SplatI64(FixedMax));
  return AddSat(MulFixed(lhs, SubSat(simd::SplatI64(FixedMax), t)), MulFixed(rhs, t));
}
[[nodiscard]] inline simd::I64x ClampLerp(const simd::I64x lhs, const simd::I64x rhs, const simd::I64x amount) noexcept {
  return Lerp(lhs, rhs, Clamp(amount, simd::SplatI64(0), simd::SplatI64(FixedMax)));
}
[[nodiscard]] inline simd::I64x SmoothStep(const simd::I64x amount) noexcept {
  const simd::I64x t = Clamp(amount, simd::SplatI64(0), simd::SplatI64(FixedMax));
  const simd::I64x square = MulUnit(t, t);
  const simd::I64x cube = MulUnit(square, t);
  return SubSat(AddSat(square, AddSat(square, square)), AddSat(cube, cube));
}
[[nodiscard]] inline simd::I64x Hermite(const simd::I64x p0,
                                        const simd::I64x m0,
                                        const simd::I64x p1,
                                        const simd::I64x m1,
                                        const simd::I64x amount) noexcept {
  const simd::I64x t = Clamp(amount, simd::SplatI64(0), simd::SplatI64(FixedMax));
  const simd::I64x t2 = MulUnit(t, t);
  const simd::I64x t3 = MulUnit(t2, t);
  const simd::I64x two_t3 = AddSat(t3, t3);
  const simd::I64x three_t2 = AddSat(t2, AddSat(t2, t2));
  const simd::I64x h00 = AddSat(SubSat(two_t3, three_t2), simd::SplatI64(FixedMax));
  const simd::I64x h10 = AddSat(SubSat(t3, AddSat(t2, t2)), t);
  const simd::I64x h01 = SubSat(three_t2, two_t3);
  const simd::I64x h11 = SubSat(t3, t2);
  return AddSat(AddSat(MulFixed(p0, h00), MulFixed(m0, h10)), AddSat(MulFixed(p1, h01), MulFixed(m1, h11)));
}
}  // namespace rund::math64
