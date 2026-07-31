#pragma once

#include <math64/nonlinear/exp.hpp>

namespace rund::math64 {
namespace detail {
inline constexpr u64 LogC1 = 13306513097844322492ull;
inline constexpr u64 LogC2 = 6653256548922161246ull;
inline constexpr u64 LogC3 = 4435504365948107497ull;
inline constexpr u64 LogC4 = 3326628274461080623ull;
inline constexpr u64 LogC5 = 2661302619568864498ull;
inline constexpr u64 LogC6 = 2217752182974053749ull;
inline constexpr u64 LogC7 = 1900930442549188927ull;

[[nodiscard]] constexpr i64 ScalarLog2(const i64 value) {
  if (value <= FixedHalf) return FixedMin;
  if (value >= FixedMax) return 0;
  const i64 offset = ::rund::math64::detail::ScalarSubSat(value, FixedMax);
  i64 power = offset;
  i64 result = ::rund::math64::detail::ScalarMulFixedScaled(power, LogC1);
  power = ::rund::math64::detail::ScalarMulFixed(power, offset);
  result = ::rund::math64::detail::ScalarSubSat(result, ::rund::math64::detail::ScalarMulFixedScaled(power, LogC2));
  power = ::rund::math64::detail::ScalarMulFixed(power, offset);
  result = ::rund::math64::detail::ScalarAddSat(result, ::rund::math64::detail::ScalarMulFixedScaled(power, LogC3));
  power = ::rund::math64::detail::ScalarMulFixed(power, offset);
  result = ::rund::math64::detail::ScalarSubSat(result, ::rund::math64::detail::ScalarMulFixedScaled(power, LogC4));
  power = ::rund::math64::detail::ScalarMulFixed(power, offset);
  result = ::rund::math64::detail::ScalarAddSat(result, ::rund::math64::detail::ScalarMulFixedScaled(power, LogC5));
  power = ::rund::math64::detail::ScalarMulFixed(power, offset);
  result = ::rund::math64::detail::ScalarSubSat(result, ::rund::math64::detail::ScalarMulFixedScaled(power, LogC6));
  power = ::rund::math64::detail::ScalarMulFixed(power, offset);
  result = ::rund::math64::detail::ScalarAddSat(result, ::rund::math64::detail::ScalarMulFixedScaled(power, LogC7));
  return ::rund::math64::detail::ScalarClamp(result, FixedMin, 0);
}
}  // namespace detail

[[nodiscard]] inline simd::I64x Log2(const simd::I64x value) noexcept {
  const simd::I64x offset = SubSat(value, simd::SplatI64(FixedMax));
  simd::I64x power = offset;
  simd::I64x result = MulFixedScaled(power, simd::SplatU64(detail::LogC1));
  power = MulFixed(power, offset);
  result = SubSat(result, MulFixedScaled(power, simd::SplatU64(detail::LogC2)));
  power = MulFixed(power, offset);
  result = AddSat(result, MulFixedScaled(power, simd::SplatU64(detail::LogC3)));
  power = MulFixed(power, offset);
  result = SubSat(result, MulFixedScaled(power, simd::SplatU64(detail::LogC4)));
  power = MulFixed(power, offset);
  result = AddSat(result, MulFixedScaled(power, simd::SplatU64(detail::LogC5)));
  power = MulFixed(power, offset);
  result = SubSat(result, MulFixedScaled(power, simd::SplatU64(detail::LogC6)));
  power = MulFixed(power, offset);
  result = AddSat(result, MulFixedScaled(power, simd::SplatU64(detail::LogC7)));
  result = Clamp(result, simd::SplatI64(FixedMin), simd::SplatI64(0));
  result = simd::Select(simd::Le(value, simd::SplatI64(FixedHalf)), simd::SplatI64(FixedMin), result);
  return simd::Select(simd::Ge(value, simd::SplatI64(FixedMax)), simd::SplatI64(0), result);
}

}  // namespace rund::math64
