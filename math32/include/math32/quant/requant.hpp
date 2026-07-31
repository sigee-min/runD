#pragma once

#include <math32/quant/clamp.hpp>

namespace rund::math32::quant {
namespace detail {
[[nodiscard]] inline RequantResult RequantI32To(const simd::I32x value,
                                                const simd::I32x multiplier,
                                                const simd::U32x shift,
                                                const simd::I32x zero_point,
                                                const simd::I32x lower,
                                                const simd::I32x upper) noexcept {
  const ::rund::math32::detail::I64x4 product =
      ::rund::math32::detail::Widen64(value) *
      ::rund::math32::detail::Widen64(multiplier);
  const ::rund::math32::detail::U64x4 shift64 =
      ::rund::math32::detail::Widen64(shift);
  const ::rund::math32::detail::Mask64x4 valid = ::rund::math32::detail::Ge64(::rund::math32::detail::SplatU64x4(62u), shift64);
  const ::rund::math32::detail::Mask64x4 negative = ::rund::math32::detail::Lt64(product, ::rund::math32::detail::SplatI64x4(0));
  const ::rund::math32::detail::U64x4 magnitude = ::rund::math32::detail::AbsMagnitudeWide(product);
  const ::rund::math32::detail::U64x4 safe_shift =
      ::rund::math32::detail::Min64(shift64, ::rund::math32::detail::SplatU64x4(62u));
  const ::rund::math32::detail::Mask64x4 non_zero_shift =
      ~::rund::math32::detail::Eq64(safe_shift, ::rund::math32::detail::SplatU64x4(0));
  const ::rund::math32::detail::U64x4 rounding_shift =
      ::rund::math32::detail::Select64(
          non_zero_shift,
          safe_shift - ::rund::math32::detail::SplatU64x4(1u),
          ::rund::math32::detail::SplatU64x4(0));
  const ::rund::math32::detail::U64x4 rounding = ::rund::math32::detail::Select64(
      non_zero_shift,
      ::rund::math32::detail::SplatU64x4(1u) << rounding_shift,
      ::rund::math32::detail::SplatU64x4(0));
  const ::rund::math32::detail::U64x4 rounded = (magnitude + rounding) >> safe_shift;
  const ::rund::math32::detail::I64x4 signed_rounded =
      ::rund::math32::detail::Select64(negative,
                                       -::rund::math32::detail::Signed64(rounded),
                                       ::rund::math32::detail::Signed64(rounded));
  const simd::I32x with_zero = ::rund::math32::detail::ClampWideToI32x(
      signed_rounded + ::rund::math32::detail::Widen64(zero_point));
  return RequantResult{.value = Clamp(with_zero, lower, upper),
                       .valid_shift =
                           ::rund::math32::detail::NarrowMask(valid)};
}
}  // namespace detail
[[nodiscard]] inline RequantResult RequantI32ToI8(const simd::I32x value,
                                                  const simd::I32x multiplier,
                                                  const simd::U32x shift,
                                                  const simd::I32x zero_point) noexcept {
  return detail::RequantI32To(value, multiplier, shift, zero_point,
                              simd::SplatI32(std::numeric_limits<i8>::min()),
                              simd::SplatI32(std::numeric_limits<i8>::max()));
}
[[nodiscard]] inline RequantResult RequantI32ToI16(const simd::I32x value,
                                                   const simd::I32x multiplier,
                                                   const simd::U32x shift,
                                                   const simd::I32x zero_point) noexcept {
  return detail::RequantI32To(value, multiplier, shift, zero_point,
                              simd::SplatI32(std::numeric_limits<i16>::min()),
                              simd::SplatI32(std::numeric_limits<i16>::max()));
}
[[nodiscard]] inline RequantResult RequantI32ToU8(const simd::I32x value,
                                                  const simd::I32x multiplier,
                                                  const simd::U32x shift,
                                                  const simd::I32x zero_point) noexcept {
  return detail::RequantI32To(value, multiplier, shift, zero_point,
                              simd::SplatI32(0),
                              simd::SplatI32(std::numeric_limits<u8>::max()));
}
}  // namespace rund::math32::quant
