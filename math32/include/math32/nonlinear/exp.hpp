#pragma once

#include <math32/fixed/sqrt.hpp>

namespace rund::math32 {
namespace detail {
inline constexpr i32 ExpC1 = -1488522236;
inline constexpr i32 ExpC2 = 515882496;
inline constexpr i32 ExpC3 = -119194166;
inline constexpr i32 ExpC4 = 20654775;
inline constexpr i32 ExpC5 = -2863360;
inline constexpr i32 ExpC6 = 330788;

[[nodiscard]] constexpr i32 ScalarExp2(const i32 value) {
  if (value >= 0) return FixedMax;
  if (value == FixedMin) return FixedHalf;
  const i32 unit = ::rund::math32::detail::ScalarAbs(value);
  i32 polynomial = ExpC6;
  polynomial = ::rund::math32::detail::ScalarAddSat(ExpC5, ::rund::math32::detail::ScalarMulFixed(polynomial, unit));
  polynomial = ::rund::math32::detail::ScalarAddSat(ExpC4, ::rund::math32::detail::ScalarMulFixed(polynomial, unit));
  polynomial = ::rund::math32::detail::ScalarAddSat(ExpC3, ::rund::math32::detail::ScalarMulFixed(polynomial, unit));
  polynomial = ::rund::math32::detail::ScalarAddSat(ExpC2, ::rund::math32::detail::ScalarMulFixed(polynomial, unit));
  polynomial = ::rund::math32::detail::ScalarAddSat(ExpC1, ::rund::math32::detail::ScalarMulFixed(polynomial, unit));
  return ::rund::math32::detail::ScalarClamp(::rund::math32::detail::ScalarAddSat(FixedMax, ::rund::math32::detail::ScalarMulFixed(polynomial, unit)), FixedHalf, FixedMax);
}
}  // namespace detail

[[nodiscard]] inline simd::I32x Exp2(const simd::I32x value) noexcept {
  const simd::I32x unit = Abs(value);
  simd::I32x polynomial = simd::SplatI32(detail::ExpC6);
  polynomial = AddSat(simd::SplatI32(detail::ExpC5), MulFixed(polynomial, unit));
  polynomial = AddSat(simd::SplatI32(detail::ExpC4), MulFixed(polynomial, unit));
  polynomial = AddSat(simd::SplatI32(detail::ExpC3), MulFixed(polynomial, unit));
  polynomial = AddSat(simd::SplatI32(detail::ExpC2), MulFixed(polynomial, unit));
  polynomial = AddSat(simd::SplatI32(detail::ExpC1), MulFixed(polynomial, unit));
  const simd::I32x approximated = Clamp(AddSat(simd::SplatI32(FixedMax), MulFixed(polynomial, unit)),
                                        simd::SplatI32(FixedHalf),
                                        simd::SplatI32(FixedMax));
  const simd::I32x non_negative = simd::Select(simd::Ge(value, simd::SplatI32(0)), simd::SplatI32(FixedMax), approximated);
  return simd::Select(simd::Eq(value, simd::SplatI32(FixedMin)), simd::SplatI32(FixedHalf), non_negative);
}

}  // namespace rund::math32
