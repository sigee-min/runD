#pragma once

#include <math32/nn/dot.hpp>

namespace rund::math32::nn {
namespace detail {
inline constexpr u64 TanhNumeratorX2 = 275318416ull;
inline constexpr u64 TanhNumeratorX4 = 6006947ull;
inline constexpr u64 TanhDenominatorX2 = 991146299ull;
inline constexpr u64 TanhDenominatorX4 = 50057894ull;
inline constexpr u64 TanhDenominatorX6 = 444959ull;
}  // namespace detail
[[nodiscard]] inline simd::I32x Relu(const simd::I32x value) noexcept {
  return ::rund::math32::Max(value, simd::SplatI32(0));
}
[[nodiscard]] inline simd::I32x TanhApprox(const simd::I32x value) noexcept {
  const simd::Mask32x min_mask = simd::Eq(value, simd::SplatI32(FixedMin));
  const simd::U32x magnitude = simd::Select(min_mask, simd::SplatU32(static_cast<u32>(FixedMax)), AbsMagnitude(value));
  const simd::U32x square = MulUnsignedFixed(magnitude, magnitude);
  const simd::U32x square2 = MulUnsignedFixed(square, square);
  const simd::U32x square3 = MulUnsignedFixed(square2, square);
  simd::U32x numerator = AddWrapUnsigned(simd::SplatU32(static_cast<u32>(FixedScale)),
                                         MulUnsignedFixed(square, simd::SplatU32(static_cast<u32>(detail::TanhNumeratorX2))));
  numerator = AddWrapUnsigned(numerator, MulUnsignedFixed(square2, simd::SplatU32(static_cast<u32>(detail::TanhNumeratorX4))));
  simd::U32x denominator = AddWrapUnsigned(simd::SplatU32(static_cast<u32>(FixedScale)),
                                           MulUnsignedFixed(square, simd::SplatU32(static_cast<u32>(detail::TanhDenominatorX2))));
  denominator = AddWrapUnsigned(denominator, MulUnsignedFixed(square2, simd::SplatU32(static_cast<u32>(detail::TanhDenominatorX4))));
  denominator = AddWrapUnsigned(denominator, MulUnsignedFixed(square3, simd::SplatU32(static_cast<u32>(detail::TanhDenominatorX6))));
  const ::rund::math32::detail::U64x4 ratio =
      ::rund::math32::detail::UnsignedDiv64(
          ::rund::math32::detail::Widen64(numerator) << 31u,
          ::rund::math32::detail::Widen64(denominator));
  const ::rund::math32::detail::U64x4 tanh_mag =
      ::rund::math32::detail::Widen64(
          MulUnsignedFixed(
              magnitude, ::rund::math32::detail::Narrow32(ratio)));
  const ::rund::math32::detail::Mask64x4 negative =
      ::rund::math32::detail::Lt64(
                                   ::rund::math32::detail::Widen64(value),
                                   ::rund::math32::detail::SplatI64x4(0));
  return ::rund::math32::detail::ClampSignedMagnitude64(negative, tanh_mag);
}
[[nodiscard]] inline simd::I32x SigmoidApprox(const simd::I32x value) noexcept {
  return AddSat(simd::SplatI32(FixedHalf), TanhApprox(value >> 1u) >> 1u);
}
}  // namespace rund::math32::nn
