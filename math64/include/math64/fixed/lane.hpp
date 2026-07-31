#pragma once

#include <math64/fixed/constants.hpp>
#include <math64/fixed/wide/lane.hpp>
#include <math64/simd/arithmetic.hpp>
#include <math64/simd/compare.hpp>
#include <math64/simd/select.hpp>

#include <bit>
#include <limits>

namespace rund::math64 {
namespace detail {
[[nodiscard]] inline simd::U64x MulWideU32(const simd::detail::U32x2 lhs,
                                           const simd::detail::U32x2 rhs) noexcept {
  return __builtin_convertvector(lhs, simd::U64x) * __builtin_convertvector(rhs, simd::U64x);
}

[[nodiscard]] inline simd::U64x MulUnsignedHigh64(const simd::U64x lhs,
                                                  const simd::U64x rhs) noexcept {
  const simd::U64x mask32 = simd::SplatU64(0xffffffffull);
  const simd::U64x p0 = MulWideU32(simd::detail::LowWords(lhs), simd::detail::LowWords(rhs));
  const simd::U64x p1 = MulWideU32(simd::detail::LowWords(lhs), simd::detail::HighWords(rhs));
  const simd::U64x p2 = MulWideU32(simd::detail::HighWords(lhs), simd::detail::LowWords(rhs));
  const simd::U64x p3 = MulWideU32(simd::detail::HighWords(lhs), simd::detail::HighWords(rhs));
  const simd::U64x carry = (p0 >> 32u) + (p1 & mask32) + (p2 & mask32);
  return p3 + (p1 >> 32u) + (p2 >> 32u) + (carry >> 32u);
}

[[nodiscard]] inline simd::U64x MulUnsignedShift63(const simd::U64x lhs,
                                                   const simd::U64x rhs) noexcept {
  const simd::U64x mask32 = simd::SplatU64(0xffffffffull);
  const simd::U64x p0 = MulWideU32(simd::detail::LowWords(lhs), simd::detail::LowWords(rhs));
  const simd::U64x p1 = MulWideU32(simd::detail::LowWords(lhs), simd::detail::HighWords(rhs));
  const simd::U64x p2 = MulWideU32(simd::detail::HighWords(lhs), simd::detail::LowWords(rhs));
  const simd::U64x p3 = MulWideU32(simd::detail::HighWords(lhs), simd::detail::HighWords(rhs));
  const simd::U64x carry = (p0 >> 32u) + (p1 & mask32) + (p2 & mask32);
  const simd::U64x high = p3 + (p1 >> 32u) + (p2 >> 32u) + (carry >> 32u);
  return (high << 1u) | ((carry >> 31u) & simd::SplatU64(1u));
}

[[nodiscard]] inline simd::I64x ClampSignedMagnitude64(const simd::Mask64x negative,
                                                       const simd::U64x magnitude) noexcept {
  const simd::U64x fixed_min_magnitude = simd::SplatU64(u64{1} << 63u);
  const simd::U64x fixed_max_magnitude = simd::SplatU64(static_cast<u64>(FixedMax));
  const simd::I64x positive = std::bit_cast<simd::I64x>(simd::Min(magnitude, fixed_max_magnitude));
  const simd::I64x negative_value =
      simd::Select(simd::Ge(magnitude, fixed_min_magnitude),
                   simd::SplatI64(FixedMin),
                   simd::Sub(simd::SplatI64(0), std::bit_cast<simd::I64x>(magnitude)));
  return simd::Select(negative, negative_value, positive);
}

}  // namespace detail

[[nodiscard]] inline simd::I64x Abs(const simd::I64x value) noexcept {
  const simd::I64x zero = simd::SplatI64(0);
  const simd::I64x wrapped = simd::Select(simd::Lt(value, zero), simd::Sub(zero, value), value);
  return simd::Select(simd::Eq(value, simd::SplatI64(FixedMin)), simd::SplatI64(FixedMax), wrapped);
}
[[nodiscard]] inline simd::U64x AbsMagnitude(const simd::I64x value) noexcept {
  const simd::I64x zero = simd::SplatI64(0);
  return std::bit_cast<simd::U64x>(simd::Select(simd::Lt(value, zero), simd::Sub(zero, value), value));
}
[[nodiscard]] inline simd::I64x Sign(const simd::I64x value) noexcept {
  const simd::I64x zero = simd::SplatI64(0);
  const simd::I64x positive = simd::Select(simd::Gt(value, zero), simd::SplatI64(1), zero);
  return simd::Select(simd::Lt(value, zero), simd::SplatI64(-1), positive);
}
[[nodiscard]] inline simd::I64x AddWrap(const simd::I64x lhs, const simd::I64x rhs) noexcept { return simd::Add(lhs, rhs); }
[[nodiscard]] inline simd::U64x AddWrapUnsigned(const simd::U64x lhs, const simd::U64x rhs) noexcept { return simd::Add(lhs, rhs); }
[[nodiscard]] inline simd::I64x SubWrap(const simd::I64x lhs, const simd::I64x rhs) noexcept { return simd::Sub(lhs, rhs); }
[[nodiscard]] inline simd::I64x MulLow(const simd::I64x lhs, const simd::I64x rhs) noexcept { return simd::MulLow(lhs, rhs); }
[[nodiscard]] inline simd::I64x MulHigh(const simd::I64x lhs, const simd::I64x rhs) noexcept {
  return std::bit_cast<simd::I64x>(detail::MulUnsignedHigh64(std::bit_cast<simd::U64x>(lhs),
                                                             std::bit_cast<simd::U64x>(rhs)));
}
[[nodiscard]] inline simd::I64x AddSat(const simd::I64x lhs, const simd::I64x rhs) noexcept {
  return detail::ClampI128x2ToI64x(detail::Widen128(lhs) +
                                   detail::Widen128(rhs));
}
[[nodiscard]] inline simd::U64x AddSatUnsigned(const simd::U64x lhs, const simd::U64x rhs) noexcept {
  const simd::U64x sum = AddWrapUnsigned(lhs, rhs);
  return simd::Select(simd::Lt(sum, lhs), simd::SplatU64(std::numeric_limits<u64>::max()), sum);
}
[[nodiscard]] inline simd::I64x SubSat(const simd::I64x lhs, const simd::I64x rhs) noexcept {
  return detail::ClampI128x2ToI64x(detail::Widen128(lhs) -
                                   detail::Widen128(rhs));
}
[[nodiscard]] inline simd::I64x Min(const simd::I64x lhs, const simd::I64x rhs) noexcept { return simd::Min(lhs, rhs); }
[[nodiscard]] inline simd::I64x Max(const simd::I64x lhs, const simd::I64x rhs) noexcept { return simd::Max(lhs, rhs); }
[[nodiscard]] inline simd::I64x Clamp(const simd::I64x value, const simd::I64x lower, const simd::I64x upper) noexcept {
  return Min(Max(value, lower), upper);
}
[[nodiscard]] inline simd::I64x Select(const simd::Mask64x condition,
                                       const simd::I64x when_true,
                                       const simd::I64x when_false) noexcept {
  return simd::Select(condition, when_true, when_false);
}
[[nodiscard]] inline simd::I64x NegPositiveFixed(const simd::I64x value) noexcept {
  const simd::I64x negated = simd::Sub(simd::SplatI64(0), value);
  return simd::Select(simd::Eq(value, simd::SplatI64(FixedMax)), simd::SplatI64(FixedMin), negated);
}
[[nodiscard]] inline simd::I64x MulFixed(const simd::I64x lhs, const simd::I64x rhs) noexcept {
  const simd::Mask64x negative = simd::Lt(lhs, simd::SplatI64(0)) ^ simd::Lt(rhs, simd::SplatI64(0));
  return detail::ClampSignedMagnitude64(negative, detail::MulUnsignedShift63(AbsMagnitude(lhs), AbsMagnitude(rhs)));
}
[[nodiscard]] inline simd::I64x MulFixedScaled(const simd::I64x value, const simd::U64x scaled_coefficient) noexcept {
  return detail::ClampSignedMagnitude64(simd::Lt(value, simd::SplatI64(0)),
                                        detail::MulUnsignedShift63(AbsMagnitude(value), scaled_coefficient));
}
[[nodiscard]] inline simd::U64x MulUnsignedFixed(const simd::U64x lhs, const simd::U64x rhs) noexcept {
  return detail::MulUnsignedShift63(lhs, rhs);
}
[[nodiscard]] inline simd::I64x MulAddFixed(const simd::I64x lhs,
                                            const simd::I64x rhs,
                                            const simd::I64x addend) noexcept {
  return AddSat(MulFixed(lhs, rhs), addend);
}
[[nodiscard]] inline simd::I64x DivFixed(const simd::I64x lhs, const simd::I64x rhs) noexcept {
  const detail::U128x2 lhs_mag = detail::AbsMagnitude128(lhs);
  const detail::U128x2 rhs_mag = detail::AbsMagnitude128(rhs);
  const detail::U128x2 quotient = detail::UnsignedDiv128(lhs_mag << 63u, rhs_mag);
  const detail::I128x2 lhs_wide = detail::Widen128(lhs);
  const detail::I128x2 rhs_wide = detail::Widen128(rhs);
  const detail::Mask128x2 negative =
      detail::Lt128(lhs_wide, detail::SplatI128x2(0)) ^
      detail::Lt128(rhs_wide, detail::SplatI128x2(0));
  const simd::I64x divided = detail::ClampSignedMagnitude128(negative, quotient);
  const simd::I64x zero_rhs = simd::Select(simd::Gt(lhs, simd::SplatI64(0)),
                                           simd::SplatI64(FixedMax),
                                           simd::Select(simd::Lt(lhs, simd::SplatI64(0)), simd::SplatI64(FixedMin), simd::SplatI64(0)));
  return simd::Select(
      detail::NarrowMask(
          detail::Eq128(rhs_mag, detail::SplatU128x2(0))),
      zero_rhs, divided);
}
[[nodiscard]] inline simd::I64x Recip(const simd::I64x value) noexcept {
  const detail::U128x2 magnitude = detail::AbsMagnitude128(value);
  const detail::U128x2 quotient = detail::UnsignedDiv128(detail::SplatU128x2(detail::u128{1} << 126u), magnitude);
  const detail::Mask128x2 negative =
      detail::Lt128(detail::Widen128(value), detail::SplatI128x2(0));
  const simd::I64x recip = detail::ClampSignedMagnitude128(negative, quotient);
  return simd::Select(simd::Eq(value, simd::SplatI64(0)), simd::SplatI64(FixedMax), recip);
}

}  // namespace rund::math64
