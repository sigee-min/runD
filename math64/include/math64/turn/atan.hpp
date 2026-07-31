#pragma once

#include <math64/turn/trig.hpp>

#include <bit>

namespace rund::math64 {
namespace detail {
[[nodiscard]] constexpr i64 TurnPoly34(const i64 square, const i64 (&coef)[34]) {
  i64 acc = coef[33];
  for (int i = 32; i >= 0; --i) acc = ::rund::math64::detail::ScalarAddSat(coef[i], ::rund::math64::detail::ScalarMulFixed(acc, square));
  return acc;
}
[[nodiscard]] inline simd::I64x TurnPoly34(const simd::I64x square, const i64 (&coef)[34]) noexcept {
  simd::I64x acc = simd::SplatI64(coef[33]);
  for (int i = 32; i >= 0; --i) acc = AddSat(simd::SplatI64(coef[i]), MulFixed(acc, square));
  return acc;
}
}  // namespace detail
[[nodiscard]] inline simd::Mask64x TurnRatioLe(const simd::U64x numerator,
                                               const simd::U64x denominator,
                                               const simd::I64x ratio) noexcept {
  const detail::U128x2 left =
      detail::Widen128(numerator) * detail::SplatU128x2(FixedScale);
  const simd::U64x ratio_pos = std::bit_cast<simd::U64x>(Max(ratio, simd::SplatI64(0)));
  const detail::U128x2 right =
      detail::Widen128(denominator) * detail::Widen128(ratio_pos);
  const detail::Mask128x2 le = ~detail::Gt128(left, right);
  const simd::Mask64x non_zero = simd::Ne(denominator, simd::SplatU64(0));
  return detail::NarrowMask(le) & non_zero;
}
[[nodiscard]] inline simd::U64x TurnRatio(const simd::U64x numerator, const simd::U64x denominator) noexcept {
  const detail::U128x2 denominator128 = detail::Widen128(denominator);
  const detail::U128x2 quotient =
      detail::UnsignedDiv128(detail::Widen128(numerator) << 63u,
                             denominator128);
  const detail::U128x2 clamped = detail::Min128(quotient, detail::SplatU128x2(static_cast<u64>(FixedMax)));
  return simd::Select(simd::Eq(denominator, simd::SplatU64(0)),
                      simd::SplatU64(static_cast<u64>(FixedMax)),
                      detail::Narrow64(clamped));
}
[[nodiscard]] inline simd::I64x TurnAtanUnit(const simd::I64x ratio) noexcept {
  return MulFixed(ratio, detail::TurnPoly34(MulFixed(ratio, ratio), detail::TurnAtanCoef));
}
[[nodiscard]] inline simd::U64x TurnAtanRatio(const simd::U64x small, const simd::U64x large) noexcept {
  const simd::Mask64x zero = simd::Eq(small, simd::SplatU64(0)) | simd::Eq(large, simd::SplatU64(0));
  const simd::I64x direct = TurnAtanUnit(std::bit_cast<simd::I64x>(TurnRatio(small, large)));
  const simd::U64x delta_ratio = TurnRatio(large - small, large + small);
  const simd::I64x delta = AddSat(simd::SplatI64(static_cast<i64>(TurnEighth)),
                                  TurnAtanUnit(std::bit_cast<simd::I64x>(delta_ratio)));
  const simd::U64x selected = simd::Select(TurnRatioLe(small, large, simd::SplatI64(FixedTanTurnEighth)),
                                           std::bit_cast<simd::U64x>(direct),
                                           std::bit_cast<simd::U64x>(delta));
  return simd::Select(zero, simd::SplatU64(0), selected);
}
[[nodiscard]] inline simd::U64x TurnAtanFirstQuadrant(const simd::U64x y_abs, const simd::U64x x_abs) noexcept {
  const simd::Mask64x y_small = simd::Le(y_abs, x_abs);
  const simd::U64x small = simd::Select(y_small, y_abs, x_abs);
  const simd::U64x large = simd::Select(y_small, x_abs, y_abs);
  const simd::U64x offset = TurnAtanRatio(small, large);
  return simd::Select(y_small, offset, simd::SplatU64(TurnQuarter) - offset);
}
[[nodiscard]] inline simd::U64x TurnAtan2(const simd::I64x y, const simd::I64x x) noexcept {
  const simd::I64x zero_i = simd::SplatI64(0);
  const simd::U64x offset = TurnAtanFirstQuadrant(AbsMagnitude(y), AbsMagnitude(x));
  simd::U64x out = simd::SplatU64(0) - offset;
  out = simd::Select(simd::Lt(x, zero_i) & simd::Lt(y, zero_i), simd::SplatU64(TurnHalf) + offset, out);
  out = simd::Select(simd::Lt(x, zero_i) & simd::Gt(y, zero_i), simd::SplatU64(TurnHalf) - offset, out);
  out = simd::Select(simd::Gt(x, zero_i) & simd::Gt(y, zero_i), offset, out);
  out = simd::Select(simd::Eq(y, zero_i), simd::Select(simd::Lt(x, zero_i), simd::SplatU64(TurnHalf), simd::SplatU64(0)), out);
  out = simd::Select(simd::Eq(x, zero_i), simd::Select(simd::Gt(y, zero_i), simd::SplatU64(TurnQuarter), simd::SplatU64(TurnThreeQuarter)), out);
  return simd::Select(simd::Eq(x, zero_i) & simd::Eq(y, zero_i), simd::SplatU64(0), out);
}
}  // namespace rund::math64
