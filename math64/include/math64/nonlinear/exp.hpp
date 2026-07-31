#pragma once

#include <math64/fixed/sqrt.hpp>

namespace rund::math64 {
namespace detail {
inline constexpr i64 ExpC1 = -6393154322601327830ll;
inline constexpr i64 ExpC2 = 2215698446797868712ll;
inline constexpr i64 ExpC3 = -511935043789664227ll;
inline constexpr i64 ExpC4 = 88711583058159475ll;
inline constexpr i64 ExpC5 = -12298036735954530ll;
inline constexpr i64 ExpC6 = 1420724914991586ll;

[[nodiscard]] constexpr i64 ScalarExp2(const i64 value) {
  if (value >= 0) return FixedMax;
  if (value == FixedMin) return FixedHalf;
  const i64 unit = ::rund::math64::detail::ScalarAbs(value);
  i64 polynomial = ExpC6;
  polynomial = ::rund::math64::detail::ScalarAddSat(ExpC5, ::rund::math64::detail::ScalarMulFixed(polynomial, unit));
  polynomial = ::rund::math64::detail::ScalarAddSat(ExpC4, ::rund::math64::detail::ScalarMulFixed(polynomial, unit));
  polynomial = ::rund::math64::detail::ScalarAddSat(ExpC3, ::rund::math64::detail::ScalarMulFixed(polynomial, unit));
  polynomial = ::rund::math64::detail::ScalarAddSat(ExpC2, ::rund::math64::detail::ScalarMulFixed(polynomial, unit));
  polynomial = ::rund::math64::detail::ScalarAddSat(ExpC1, ::rund::math64::detail::ScalarMulFixed(polynomial, unit));
  return ::rund::math64::detail::ScalarClamp(::rund::math64::detail::ScalarAddSat(FixedMax, ::rund::math64::detail::ScalarMulFixed(polynomial, unit)), FixedHalf, FixedMax);
}
}  // namespace detail

[[nodiscard]] inline simd::I64x Exp2(const simd::I64x value) noexcept {
  const simd::I64x unit = Abs(value);
  simd::I64x polynomial = simd::SplatI64(detail::ExpC6);
  polynomial = AddSat(simd::SplatI64(detail::ExpC5), MulFixed(polynomial, unit));
  polynomial = AddSat(simd::SplatI64(detail::ExpC4), MulFixed(polynomial, unit));
  polynomial = AddSat(simd::SplatI64(detail::ExpC3), MulFixed(polynomial, unit));
  polynomial = AddSat(simd::SplatI64(detail::ExpC2), MulFixed(polynomial, unit));
  polynomial = AddSat(simd::SplatI64(detail::ExpC1), MulFixed(polynomial, unit));
  const simd::I64x approximated = Clamp(AddSat(simd::SplatI64(FixedMax), MulFixed(polynomial, unit)),
                                        simd::SplatI64(FixedHalf),
                                        simd::SplatI64(FixedMax));
  const simd::I64x non_negative = simd::Select(simd::Ge(value, simd::SplatI64(0)), simd::SplatI64(FixedMax), approximated);
  return simd::Select(simd::Eq(value, simd::SplatI64(FixedMin)), simd::SplatI64(FixedHalf), non_negative);
}

}  // namespace rund::math64
