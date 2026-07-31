#pragma once

#include <math32/turn/trig.hpp>

namespace rund::math32 {
namespace detail {
[[nodiscard]] constexpr i32 TurnPoly12(const i32 square, const i32 (&coef)[12]) {
  const i32 s2 = ::rund::math32::detail::ScalarMulFixed(square, square);
  const i32 s4 = ::rund::math32::detail::ScalarMulFixed(s2, s2);
  const i32 s8 = ::rund::math32::detail::ScalarMulFixed(s4, s4);
  const i32 p0 = TurnPair(coef[0], coef[1], square);
  const i32 p1 = TurnPair(coef[2], coef[3], square);
  const i32 p2 = TurnPair(coef[4], coef[5], square);
  const i32 p3 = TurnPair(coef[6], coef[7], square);
  const i32 p4 = TurnPair(coef[8], coef[9], square);
  const i32 p5 = TurnPair(coef[10], coef[11], square);
  return ::rund::math32::detail::ScalarAddSat(::rund::math32::detail::ScalarAddSat(::rund::math32::detail::ScalarAddSat(p0, ::rund::math32::detail::ScalarMulFixed(p1, s2)), ::rund::math32::detail::ScalarMulFixed(::rund::math32::detail::ScalarAddSat(p2, ::rund::math32::detail::ScalarMulFixed(p3, s2)), s4)),
                ::rund::math32::detail::ScalarMulFixed(::rund::math32::detail::ScalarAddSat(p4, ::rund::math32::detail::ScalarMulFixed(p5, s2)), s8));
}
[[nodiscard]] inline simd::I32x TurnPoly12(const simd::I32x square, const i32 (&coef)[12]) noexcept {
  const simd::I32x s2 = ::rund::math32::MulFixed(square, square);
  const simd::I32x s4 = ::rund::math32::MulFixed(s2, s2);
  const simd::I32x s8 = ::rund::math32::MulFixed(s4, s4);
  const simd::I32x p0 = TurnPair(simd::SplatI32(coef[0]), simd::SplatI32(coef[1]), square);
  const simd::I32x p1 = TurnPair(simd::SplatI32(coef[2]), simd::SplatI32(coef[3]), square);
  const simd::I32x p2 = TurnPair(simd::SplatI32(coef[4]), simd::SplatI32(coef[5]), square);
  const simd::I32x p3 = TurnPair(simd::SplatI32(coef[6]), simd::SplatI32(coef[7]), square);
  const simd::I32x p4 = TurnPair(simd::SplatI32(coef[8]), simd::SplatI32(coef[9]), square);
  const simd::I32x p5 = TurnPair(simd::SplatI32(coef[10]), simd::SplatI32(coef[11]), square);
  return AddSat(AddSat(AddSat(p0, MulFixed(p1, s2)), MulFixed(AddSat(p2, MulFixed(p3, s2)), s4)),
                MulFixed(AddSat(p4, MulFixed(p5, s2)), s8));
}
}  // namespace detail

[[nodiscard]] inline simd::Mask32x TurnRatioLe(const simd::U32x numerator,
                                               const simd::U32x denominator,
                                               const simd::I32x ratio) noexcept {
  const detail::U64x4 left =
      detail::Widen64(numerator) * detail::SplatU64x4(FixedScale);
  const simd::U32x ratio_pos = std::bit_cast<simd::U32x>(Max(ratio, simd::SplatI32(0)));
  const detail::U64x4 right =
      detail::Widen64(denominator) * detail::Widen64(ratio_pos);
  const detail::Mask64x4 le = ~detail::Gt64(left, right);
  const simd::Mask32x non_zero = simd::Ne(denominator, simd::SplatU32(0));
  return detail::NarrowMask(le) & non_zero;
}
[[nodiscard]] inline simd::U32x TurnRatio(const simd::U32x numerator, const simd::U32x denominator) noexcept {
  const detail::U64x4 denominator64 = detail::Widen64(denominator);
  const detail::U64x4 quotient =
      detail::UnsignedDiv64(detail::Widen64(numerator) << 31u,
                            denominator64);
  const simd::U32x clamped = std::bit_cast<simd::U32x>(detail::ClampU64x4ToI32x(quotient));
  return simd::Select(simd::Eq(denominator, simd::SplatU32(0)), simd::SplatU32(static_cast<u32>(FixedMax)), clamped);
}
[[nodiscard]] inline simd::I32x TurnAtanUnit(const simd::I32x ratio) noexcept {
  const simd::I32x square = MulFixed(ratio, ratio);
  return MulFixed(ratio, detail::TurnPoly12(square, detail::TurnAtanCoef));
}
[[nodiscard]] inline simd::U32x TurnAtanRatio(const simd::U32x small, const simd::U32x large) noexcept {
  const simd::Mask32x zero = simd::Eq(small, simd::SplatU32(0)) | simd::Eq(large, simd::SplatU32(0));
  const simd::I32x direct = TurnAtanUnit(std::bit_cast<simd::I32x>(TurnRatio(small, large)));
  const simd::U32x delta_ratio = TurnRatio(large - small, large + small);
  const simd::I32x delta = AddSat(simd::SplatI32(static_cast<i32>(TurnEighth)),
                                  TurnAtanUnit(std::bit_cast<simd::I32x>(delta_ratio)));
  const simd::U32x selected = simd::Select(TurnRatioLe(small, large, simd::SplatI32(FixedTanTurnEighth)),
                                           std::bit_cast<simd::U32x>(direct),
                                           std::bit_cast<simd::U32x>(delta));
  return simd::Select(zero, simd::SplatU32(0), selected);
}
[[nodiscard]] inline simd::U32x TurnAtanFirstQuadrant(const simd::U32x y_abs, const simd::U32x x_abs) noexcept {
  const simd::Mask32x y_small = simd::Le(y_abs, x_abs);
  const simd::U32x small = simd::Select(y_small, y_abs, x_abs);
  const simd::U32x large = simd::Select(y_small, x_abs, y_abs);
  const simd::U32x offset = TurnAtanRatio(small, large);
  return simd::Select(y_small, offset, simd::SplatU32(TurnQuarter) - offset);
}
[[nodiscard]] inline simd::U32x TurnAtan2(const simd::I32x y, const simd::I32x x) noexcept {
  const simd::I32x zero_i = simd::SplatI32(0);
  const simd::U32x offset = TurnAtanFirstQuadrant(AbsMagnitude(y), AbsMagnitude(x));
  simd::U32x out = simd::SplatU32(0) - offset;
  out = simd::Select(simd::Lt(x, zero_i) & simd::Lt(y, zero_i), simd::SplatU32(TurnHalf) + offset, out);
  out = simd::Select(simd::Lt(x, zero_i) & simd::Gt(y, zero_i), simd::SplatU32(TurnHalf) - offset, out);
  out = simd::Select(simd::Gt(x, zero_i) & simd::Gt(y, zero_i), offset, out);
  out = simd::Select(simd::Eq(y, zero_i), simd::Select(simd::Lt(x, zero_i), simd::SplatU32(TurnHalf), simd::SplatU32(0)), out);
  out = simd::Select(simd::Eq(x, zero_i), simd::Select(simd::Gt(y, zero_i), simd::SplatU32(TurnQuarter), simd::SplatU32(TurnThreeQuarter)), out);
  return simd::Select(simd::Eq(x, zero_i) & simd::Eq(y, zero_i), simd::SplatU32(0), out);
}
}  // namespace rund::math32
