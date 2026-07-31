#pragma once

#include <math32/nonlinear/exp.hpp>

namespace rund::math32 {
namespace detail {
inline constexpr u64 LogC1 = 3098164009ull;
inline constexpr u64 LogC2 = 1549082005ull;
inline constexpr u64 LogC3 = 1032721336ull;
inline constexpr u64 LogC4 = 774541002ull;
inline constexpr u64 LogC5 = 619632802ull;
inline constexpr u64 LogC6 = 516360668ull;
inline constexpr u64 LogC7 = 442594858ull;

[[nodiscard]] constexpr i32 ScalarLog2(const i32 value) {
  if (value <= FixedHalf) return FixedMin;
  if (value >= FixedMax) return 0;
  const i32 offset = ::rund::math32::detail::ScalarSubSat(value, FixedMax);
  i32 power = offset;
  i32 result = ::rund::math32::detail::ScalarMulFixedScaled(power, LogC1);
  power = ::rund::math32::detail::ScalarMulFixed(power, offset);
  result = ::rund::math32::detail::ScalarSubSat(result, ::rund::math32::detail::ScalarMulFixedScaled(power, LogC2));
  power = ::rund::math32::detail::ScalarMulFixed(power, offset);
  result = ::rund::math32::detail::ScalarAddSat(result, ::rund::math32::detail::ScalarMulFixedScaled(power, LogC3));
  power = ::rund::math32::detail::ScalarMulFixed(power, offset);
  result = ::rund::math32::detail::ScalarSubSat(result, ::rund::math32::detail::ScalarMulFixedScaled(power, LogC4));
  power = ::rund::math32::detail::ScalarMulFixed(power, offset);
  result = ::rund::math32::detail::ScalarAddSat(result, ::rund::math32::detail::ScalarMulFixedScaled(power, LogC5));
  power = ::rund::math32::detail::ScalarMulFixed(power, offset);
  result = ::rund::math32::detail::ScalarSubSat(result, ::rund::math32::detail::ScalarMulFixedScaled(power, LogC6));
  power = ::rund::math32::detail::ScalarMulFixed(power, offset);
  result = ::rund::math32::detail::ScalarAddSat(result, ::rund::math32::detail::ScalarMulFixedScaled(power, LogC7));
  return ::rund::math32::detail::ScalarClamp(result, FixedMin, 0);
}
}  // namespace detail

[[nodiscard]] inline simd::I32x Log2(const simd::I32x value) noexcept {
  const simd::I32x offset = SubSat(value, simd::SplatI32(FixedMax));
  simd::I32x power = offset;
  simd::I32x result = MulFixedScaled(power, simd::SplatU32(static_cast<u32>(detail::LogC1)));
  power = MulFixed(power, offset);
  result = SubSat(result, MulFixedScaled(power, simd::SplatU32(static_cast<u32>(detail::LogC2))));
  power = MulFixed(power, offset);
  result = AddSat(result, MulFixedScaled(power, simd::SplatU32(static_cast<u32>(detail::LogC3))));
  power = MulFixed(power, offset);
  result = SubSat(result, MulFixedScaled(power, simd::SplatU32(static_cast<u32>(detail::LogC4))));
  power = MulFixed(power, offset);
  result = AddSat(result, MulFixedScaled(power, simd::SplatU32(static_cast<u32>(detail::LogC5))));
  power = MulFixed(power, offset);
  result = SubSat(result, MulFixedScaled(power, simd::SplatU32(static_cast<u32>(detail::LogC6))));
  power = MulFixed(power, offset);
  result = AddSat(result, MulFixedScaled(power, simd::SplatU32(static_cast<u32>(detail::LogC7))));
  result = Clamp(result, simd::SplatI32(FixedMin), simd::SplatI32(0));
  result = simd::Select(simd::Le(value, simd::SplatI32(FixedHalf)), simd::SplatI32(FixedMin), result);
  return simd::Select(simd::Ge(value, simd::SplatI32(FixedMax)), simd::SplatI32(0), result);
}

}  // namespace rund::math32
