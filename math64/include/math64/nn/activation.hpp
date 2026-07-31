#pragma once

#include <math64/nn/dot.hpp>

namespace rund::math64::nn {
namespace detail {
inline constexpr u64 TanhNumeratorX2 = 1182483592706523136ull;
inline constexpr u64 TanhNumeratorX4 = 25799640913805312ull;
inline constexpr u64 TanhDenominatorX2 = 4256940939756437504ull;
inline constexpr u64 TanhDenominatorX4 = 214997017636634624ull;
inline constexpr u64 TanhDenominatorX6 = 1911084353060864ull;
}  // namespace detail
[[nodiscard]] inline simd::I64x Relu(const simd::I64x value) noexcept {
  return ::rund::math64::Max(value, simd::SplatI64(0));
}
[[nodiscard]] inline simd::I64x TanhApprox(const simd::I64x value) noexcept {
  const simd::Mask64x min_mask = simd::Eq(value, simd::SplatI64(FixedMin));
  const simd::U64x magnitude = simd::Select(min_mask, simd::SplatU64(static_cast<u64>(FixedMax)), AbsMagnitude(value));
  const simd::U64x square = MulUnsignedFixed(magnitude, magnitude);
  const simd::U64x square2 = MulUnsignedFixed(square, square);
  const simd::U64x square3 = MulUnsignedFixed(square2, square);
  simd::U64x numerator = AddWrapUnsigned(simd::SplatU64(static_cast<u64>(FixedScale)),
                                         MulUnsignedFixed(square, simd::SplatU64(static_cast<u64>(detail::TanhNumeratorX2))));
  numerator = AddWrapUnsigned(numerator, MulUnsignedFixed(square2, simd::SplatU64(static_cast<u64>(detail::TanhNumeratorX4))));
  simd::U64x denominator = AddWrapUnsigned(simd::SplatU64(static_cast<u64>(FixedScale)),
                                           MulUnsignedFixed(square, simd::SplatU64(static_cast<u64>(detail::TanhDenominatorX2))));
  denominator = AddWrapUnsigned(denominator, MulUnsignedFixed(square2, simd::SplatU64(static_cast<u64>(detail::TanhDenominatorX4))));
  denominator = AddWrapUnsigned(denominator, MulUnsignedFixed(square3, simd::SplatU64(static_cast<u64>(detail::TanhDenominatorX6))));
  const ::rund::math64::detail::U128x2 ratio =
      ::rund::math64::detail::UnsignedDiv128(
          ::rund::math64::detail::Widen128(numerator) << 63u,
          ::rund::math64::detail::Widen128(denominator));
  const simd::U64x tanh_mag =
      MulUnsignedFixed(magnitude,
                       ::rund::math64::detail::Narrow64(ratio));
  return ::rund::math64::detail::ClampSignedMagnitude64(simd::Lt(value, simd::SplatI64(0)), tanh_mag);
}
[[nodiscard]] inline simd::I64x SigmoidApprox(const simd::I64x value) noexcept {
  return AddSat(simd::SplatI64(FixedHalf), TanhApprox(value >> 1u) >> 1u);
}
}  // namespace rund::math64::nn
