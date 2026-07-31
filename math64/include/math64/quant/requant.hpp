#pragma once

#include <math64/quant/clamp.hpp>

#include <bit>

namespace rund::math64::quant {
namespace detail {
[[nodiscard]] inline RequantResult RequantI64To(const simd::I64x value,
                                                const simd::I64x multiplier,
                                                const simd::U64x shift,
                                                const simd::I64x zero_point,
                                                const simd::I64x lower,
                                                const simd::I64x upper) noexcept {
  const ::rund::math64::detail::I128x2 product =
      ::rund::math64::detail::Widen128(value) *
      ::rund::math64::detail::Widen128(multiplier);
  const ::rund::math64::detail::U128x2 shift128 =
      ::rund::math64::detail::Widen128(shift);
  const ::rund::math64::detail::Mask128x2 valid =
      ::rund::math64::detail::Ge128(::rund::math64::detail::SplatU128x2(126u), shift128);
  const ::rund::math64::detail::Mask128x2 negative =
      ::rund::math64::detail::Lt128(
          product, ::rund::math64::detail::SplatI128x2(0));
  const ::rund::math64::detail::U128x2 magnitude =
      ::rund::math64::detail::Unsigned128(
          ::rund::math64::detail::Select128(
              negative, -product, product));
  const ::rund::math64::detail::U128x2 safe_shift =
      ::rund::math64::detail::Min128(shift128, ::rund::math64::detail::SplatU128x2(126u));
  const ::rund::math64::detail::Mask128x2 non_zero_shift =
      ~::rund::math64::detail::Eq128(safe_shift, ::rund::math64::detail::SplatU128x2(0));
  const ::rund::math64::detail::U128x2 rounding_shift =
      ::rund::math64::detail::Select128(
          non_zero_shift,
          safe_shift - ::rund::math64::detail::SplatU128x2(1u),
          ::rund::math64::detail::SplatU128x2(0));
  const ::rund::math64::detail::U128x2 rounding = ::rund::math64::detail::Select128(
      non_zero_shift,
      ::rund::math64::detail::SplatU128x2(1u) << rounding_shift,
      ::rund::math64::detail::SplatU128x2(0));
  const ::rund::math64::detail::U128x2 rounded = (magnitude + rounding) >> safe_shift;
  const ::rund::math64::detail::I128x2 rounded_signed =
      ::rund::math64::detail::Signed128(rounded);
  const ::rund::math64::detail::I128x2 signed_rounded =
      ::rund::math64::detail::Select128(negative, -rounded_signed, rounded_signed);
  const simd::I64x with_zero = ::rund::math64::detail::ClampI128x2ToI64x(
      signed_rounded + ::rund::math64::detail::Widen128(zero_point));
  return RequantResult{.value = Clamp(with_zero, lower, upper),
                       .valid_shift =
                           ::rund::math64::detail::NarrowMask(valid)};
}
}  // namespace detail
[[nodiscard]] inline RequantResult RequantI64ToI8(const simd::I64x value,
                                                  const simd::I64x multiplier,
                                                  const simd::U64x shift,
                                                  const simd::I64x zero_point) noexcept {
  return detail::RequantI64To(value, multiplier, shift, zero_point,
                              simd::SplatI64(std::numeric_limits<i8>::min()),
                              simd::SplatI64(std::numeric_limits<i8>::max()));
}
[[nodiscard]] inline RequantResult RequantI64ToI16(const simd::I64x value,
                                                   const simd::I64x multiplier,
                                                   const simd::U64x shift,
                                                   const simd::I64x zero_point) noexcept {
  return detail::RequantI64To(value, multiplier, shift, zero_point,
                              simd::SplatI64(std::numeric_limits<i16>::min()),
                              simd::SplatI64(std::numeric_limits<i16>::max()));
}
[[nodiscard]] inline RequantResult RequantI64ToU8(const simd::I64x value,
                                                  const simd::I64x multiplier,
                                                  const simd::U64x shift,
                                                  const simd::I64x zero_point) noexcept {
  return detail::RequantI64To(value, multiplier, shift, zero_point,
                              simd::SplatI64(0),
                              simd::SplatI64(std::numeric_limits<u8>::max()));
}
}  // namespace rund::math64::quant
