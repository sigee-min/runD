#pragma once

#include <math32/fixed/constants.hpp>
#include <math32/fixed/wide/lane.hpp>
#include <math32/simd/arithmetic.hpp>
#include <math32/simd/compare.hpp>
#include <math32/simd/select.hpp>

#include <bit>
#include <limits>

namespace rund::math32 {
namespace detail {
[[nodiscard]] inline I64x4 DivFixedScale(I64x4 product) noexcept {
  const I64x4 shifted = product >> 31u;
  const U64x4 remainder = std::bit_cast<U64x4>(product) & SplatU64x4((u64{1} << 31u) - 1u);
  const Mask64x4 correction_mask = Lt64(product, SplatI64x4(0)) & ~Eq64(remainder, SplatU64x4(0));
  return shifted + Select64(correction_mask, SplatI64x4(1), SplatI64x4(0));
}
}  // namespace detail

[[nodiscard]] inline simd::I32x Abs(const simd::I32x value) noexcept {
  const simd::I32x zero = simd::SplatI32(0);
  const simd::I32x wrapped = simd::Select(simd::Lt(value, zero), simd::Sub(zero, value), value);
  return simd::Select(simd::Eq(value, simd::SplatI32(FixedMin)), simd::SplatI32(FixedMax), wrapped);
}
[[nodiscard]] inline simd::U32x AbsMagnitude(const simd::I32x value) noexcept {
  const simd::I32x zero = simd::SplatI32(0);
  return std::bit_cast<simd::U32x>(simd::Select(simd::Lt(value, zero), simd::Sub(zero, value), value));
}
[[nodiscard]] inline simd::I32x Sign(const simd::I32x value) noexcept {
  const simd::I32x zero = simd::SplatI32(0);
  const simd::I32x positive = simd::Select(simd::Gt(value, zero), simd::SplatI32(1), zero);
  return simd::Select(simd::Lt(value, zero), simd::SplatI32(-1), positive);
}
[[nodiscard]] inline simd::I32x AddWrap(const simd::I32x lhs, const simd::I32x rhs) noexcept {
  return simd::Add(lhs, rhs);
}
[[nodiscard]] inline simd::U32x AddWrapUnsigned(const simd::U32x lhs, const simd::U32x rhs) noexcept {
  return simd::Add(lhs, rhs);
}
[[nodiscard]] inline simd::I32x SubWrap(const simd::I32x lhs, const simd::I32x rhs) noexcept {
  return simd::Sub(lhs, rhs);
}
[[nodiscard]] inline simd::I32x MulLow(const simd::I32x lhs, const simd::I32x rhs) noexcept {
  return simd::MulLow(lhs, rhs);
}
[[nodiscard]] inline simd::I32x MulHigh(const simd::I32x lhs, const simd::I32x rhs) noexcept {
  const simd::U32x lhs_bits = std::bit_cast<simd::U32x>(lhs);
  const simd::U32x rhs_bits = std::bit_cast<simd::U32x>(rhs);
  const simd::U32x lo_mask = simd::SplatU32(0xffffu);
  const simd::U32x p0 = (lhs_bits & lo_mask) * (rhs_bits & lo_mask);
  const simd::U32x p1 = (lhs_bits & lo_mask) * (rhs_bits >> 16u);
  const simd::U32x p2 = (lhs_bits >> 16u) * (rhs_bits & lo_mask);
  const simd::U32x p3 = (lhs_bits >> 16u) * (rhs_bits >> 16u);
  const simd::U32x middle = (p0 >> 16u) + (p1 & lo_mask) + (p2 & lo_mask);
  return std::bit_cast<simd::I32x>(p3 + (p1 >> 16u) + (p2 >> 16u) + (middle >> 16u));
}
[[nodiscard]] inline simd::I32x AddSat(const simd::I32x lhs, const simd::I32x rhs) noexcept {
  return detail::ClampWideToI32x(detail::Widen64(lhs) +
                                 detail::Widen64(rhs));
}
[[nodiscard]] inline simd::U32x AddSatUnsigned(const simd::U32x lhs, const simd::U32x rhs) noexcept {
  const simd::U32x sum = AddWrapUnsigned(lhs, rhs);
  return simd::Select(simd::Lt(sum, lhs), simd::SplatU32(std::numeric_limits<u32>::max()), sum);
}
[[nodiscard]] inline simd::I32x SubSat(const simd::I32x lhs, const simd::I32x rhs) noexcept {
  return detail::ClampWideToI32x(detail::Widen64(lhs) -
                                 detail::Widen64(rhs));
}
[[nodiscard]] inline simd::I32x Min(const simd::I32x lhs, const simd::I32x rhs) noexcept { return simd::Min(lhs, rhs); }
[[nodiscard]] inline simd::I32x Max(const simd::I32x lhs, const simd::I32x rhs) noexcept { return simd::Max(lhs, rhs); }
[[nodiscard]] inline simd::I32x Clamp(const simd::I32x value, const simd::I32x lower, const simd::I32x upper) noexcept {
  return Min(Max(value, lower), upper);
}
[[nodiscard]] inline simd::I32x Select(const simd::Mask32x condition,
                                       const simd::I32x when_true,
                                       const simd::I32x when_false) noexcept {
  return simd::Select(condition, when_true, when_false);
}
[[nodiscard]] inline simd::I32x NegPositiveFixed(const simd::I32x value) noexcept {
  const simd::I32x negated = simd::Sub(simd::SplatI32(0), value);
  return simd::Select(simd::Eq(value, simd::SplatI32(FixedMax)), simd::SplatI32(FixedMin), negated);
}
[[nodiscard]] inline simd::I32x MulFixed(const simd::I32x lhs, const simd::I32x rhs) noexcept {
  const detail::I64x4 product =
      detail::Widen64(lhs) * detail::Widen64(rhs);
  return detail::ClampWideToI32x(detail::DivFixedScale(product));
}
[[nodiscard]] inline simd::I32x MulFixedScaled(const simd::I32x value, const simd::U32x scaled_coefficient) noexcept {
  const detail::I64x4 product =
      detail::Widen64(value) *
      detail::Signed64(detail::Widen64(scaled_coefficient));
  return detail::ClampWideToI32x(detail::DivFixedScale(product));
}
[[nodiscard]] inline simd::U32x MulUnsignedFixed(const simd::U32x lhs, const simd::U32x rhs) noexcept {
  const detail::U64x4 product =
      detail::Widen64(lhs) * detail::Widen64(rhs);
  return detail::Narrow32(product >> 31u);
}
[[nodiscard]] inline simd::I32x MulAddFixed(const simd::I32x lhs,
                                            const simd::I32x rhs,
                                            const simd::I32x addend) noexcept {
  return AddSat(MulFixed(lhs, rhs), addend);
}
[[nodiscard]] inline simd::I32x DivFixed(const simd::I32x lhs, const simd::I32x rhs) noexcept {
  const detail::U64x4 lhs_mag = detail::AbsMagnitudeWide(lhs);
  const detail::U64x4 rhs_mag = detail::AbsMagnitudeWide(rhs);
  const detail::U64x4 quotient = detail::UnsignedDiv64(lhs_mag << 31u, rhs_mag);
  const detail::I64x4 lhs_wide = detail::Widen64(lhs);
  const detail::I64x4 rhs_wide = detail::Widen64(rhs);
  const detail::Mask64x4 negative =
      detail::Lt64(lhs_wide, detail::SplatI64x4(0)) ^
      detail::Lt64(rhs_wide, detail::SplatI64x4(0));
  const simd::I32x divided = detail::ClampSignedMagnitude64(negative, quotient);
  const simd::I32x zero_rhs = simd::Select(simd::Gt(lhs, simd::SplatI32(0)),
                                           simd::SplatI32(FixedMax),
                                           simd::Select(simd::Lt(lhs, simd::SplatI32(0)),
                                                        simd::SplatI32(FixedMin),
                                                        simd::SplatI32(0)));
  return simd::Select(detail::NarrowMask(
                          detail::Eq64(rhs_mag, detail::SplatU64x4(0))),
                      zero_rhs,
                      divided);
}
[[nodiscard]] inline simd::I32x Recip(const simd::I32x value) noexcept {
  const detail::U64x4 magnitude = detail::AbsMagnitudeWide(value);
  const detail::U64x4 quotient = detail::UnsignedDiv64(detail::SplatU64x4(u64{1} << 62u), magnitude);
  const detail::Mask64x4 negative =
      detail::Lt64(detail::Widen64(value), detail::SplatI64x4(0));
  const simd::I32x recip = detail::ClampSignedMagnitude64(negative, quotient);
  return simd::Select(simd::Eq(value, simd::SplatI32(0)), simd::SplatI32(FixedMax), recip);
}

}  // namespace rund::math32
