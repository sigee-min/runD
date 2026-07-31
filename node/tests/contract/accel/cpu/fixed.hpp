#pragma once

#include "local.hpp"

#include <bit>
#include <limits>
#include <type_traits>

namespace node_accel_contract::cpu::fixed {

template <typename T>
[[nodiscard]] constexpr T DivNearestEven(
    T lhs, T rhs,
    unsigned fraction = sizeof(T) * 8u - 1u) noexcept;

template <typename T>
[[nodiscard]] constexpr T QuantizeProduct(
    T lhs, T rhs,
    unsigned fraction = sizeof(T) * 8u - 1u) noexcept;

[[nodiscard]] inline rund::kernel::i32 Max32() noexcept { return 0x7fffffff; }

[[nodiscard]] inline rund::kernel::i64 Max64() noexcept {
  return 0x7fffffffffffffffll;
}

[[nodiscard]] inline rund::kernel::i32
Saturate32(const rund::kernel::i32 value) noexcept {
  return rund::math32::detail::ScalarClamp(value, 0, Max32());
}

[[nodiscard]] inline rund::kernel::i64
Saturate64(const rund::kernel::i64 value) noexcept {
  return rund::math64::detail::ScalarClamp(value, 0, Max64());
}

[[nodiscard]] inline rund::kernel::i32
Lerp32(const rund::kernel::i32 lhs, const rund::kernel::i32 rhs,
       const rund::kernel::i32 amount) noexcept {
  const rund::kernel::i32 t = Saturate32(amount);
  return rund::math32::detail::ScalarAddSat(
      QuantizeProduct(lhs, rund::math32::detail::ScalarSubSat(Max32(), t)),
      QuantizeProduct(rhs, t));
}

[[nodiscard]] inline rund::kernel::i64
Lerp64(const rund::kernel::i64 lhs, const rund::kernel::i64 rhs,
       const rund::kernel::i64 amount) noexcept {
  const rund::kernel::i64 t = Saturate64(amount);
  return rund::math64::detail::ScalarAddSat(
      QuantizeProduct(lhs, rund::math64::detail::ScalarSubSat(Max64(), t)),
      QuantizeProduct(rhs, t));
}

[[nodiscard]] inline rund::kernel::i32
Fade32(const rund::kernel::i32 amount) noexcept {
  const rund::kernel::i32 t = Saturate32(amount);
  const rund::kernel::i32 t2 = QuantizeProduct(t, t);
  const rund::kernel::i32 inv = rund::math32::detail::ScalarSubSat(Max32(), t);
  const rund::kernel::i32 bump =
      QuantizeProduct(t2, inv);
  return rund::math32::detail::ScalarAddSat(
      t2, rund::math32::detail::ScalarAddSat(bump, bump));
}

[[nodiscard]] inline rund::kernel::i64
Fade64(const rund::kernel::i64 amount) noexcept {
  const rund::kernel::i64 t = Saturate64(amount);
  const rund::kernel::i64 t2 = QuantizeProduct(t, t);
  const rund::kernel::i64 inv = rund::math64::detail::ScalarSubSat(Max64(), t);
  const rund::kernel::i64 bump =
      QuantizeProduct(t2, inv);
  return rund::math64::detail::ScalarAddSat(
      t2, rund::math64::detail::ScalarAddSat(bump, bump));
}

[[nodiscard]] inline rund::kernel::i32
Unlerp32(const rund::kernel::i32 lo, const rund::kernel::i32 hi,
         const rund::kernel::i32 value) noexcept {
  if (hi <= lo) {
    return 0;
  }
  const rund::kernel::i32 span = rund::math32::detail::ScalarSubSat(hi, lo);
  return Saturate32(
      DivNearestEven(rund::math32::detail::ScalarSubSat(value, lo), span));
}

[[nodiscard]] inline rund::kernel::i64
Unlerp64(const rund::kernel::i64 lo, const rund::kernel::i64 hi,
         const rund::kernel::i64 value) noexcept {
  if (hi <= lo) {
    return 0;
  }
  const rund::kernel::i64 span = rund::math64::detail::ScalarSubSat(hi, lo);
  return Saturate64(
      DivNearestEven(rund::math64::detail::ScalarSubSat(value, lo), span));
}

[[nodiscard]] inline rund::kernel::i32
Remap32(const rund::kernel::i32 in_lo, const rund::kernel::i32 in_hi,
        const rund::kernel::i32 out_lo, const rund::kernel::i32 out_hi,
        const rund::kernel::i32 value) noexcept {
  return Lerp32(out_lo, out_hi, Unlerp32(in_lo, in_hi, value));
}

[[nodiscard]] inline rund::kernel::i64
Remap64(const rund::kernel::i64 in_lo, const rund::kernel::i64 in_hi,
        const rund::kernel::i64 out_lo, const rund::kernel::i64 out_hi,
        const rund::kernel::i64 value) noexcept {
  return Lerp64(out_lo, out_hi, Unlerp64(in_lo, in_hi, value));
}

[[nodiscard]] inline rund::kernel::i32
Smoothstep32(const rund::kernel::i32 edge0, const rund::kernel::i32 edge1,
             const rund::kernel::i32 value) noexcept {
  return Fade32(Unlerp32(edge0, edge1, value));
}

[[nodiscard]] inline rund::kernel::i64
Smoothstep64(const rund::kernel::i64 edge0, const rund::kernel::i64 edge1,
             const rund::kernel::i64 value) noexcept {
  return Fade64(Unlerp64(edge0, edge1, value));
}

[[nodiscard]] inline rund::kernel::i32
SmootherScale5_32(const rund::kernel::i32 value) noexcept {
  const rund::kernel::i32 two =
      rund::math32::detail::ScalarAddSat(value, value);
  const rund::kernel::i32 four =
      rund::math32::detail::ScalarAddSat(two, two);
  return rund::math32::detail::ScalarAddSat(four, value);
}

[[nodiscard]] inline rund::kernel::i64
SmootherScale5_64(const rund::kernel::i64 value) noexcept {
  const rund::kernel::i64 two =
      rund::math64::detail::ScalarAddSat(value, value);
  const rund::kernel::i64 four =
      rund::math64::detail::ScalarAddSat(two, two);
  return rund::math64::detail::ScalarAddSat(four, value);
}

[[nodiscard]] inline rund::kernel::i32
Smootherstep32(const rund::kernel::i32 edge0, const rund::kernel::i32 edge1,
               const rund::kernel::i32 value) noexcept {
  const rund::kernel::i32 t = Unlerp32(edge0, edge1, value);
  const rund::kernel::i32 inv = rund::math32::detail::ScalarSubSat(Max32(), t);
  const rund::kernel::i32 t2 = QuantizeProduct(t, t);
  const rund::kernel::i32 t3 = QuantizeProduct(t2, t);
  const rund::kernel::i32 t4 = QuantizeProduct(t3, t);
  const rund::kernel::i32 t5 = QuantizeProduct(t4, t);
  const rund::kernel::i32 inv2 = QuantizeProduct(inv, inv);
  const rund::kernel::i32 first = QuantizeProduct(t3, inv2);
  const rund::kernel::i32 second = QuantizeProduct(t4, inv);
  const rund::kernel::i32 ten =
      rund::math32::detail::ScalarAddSat(SmootherScale5_32(first),
                                         SmootherScale5_32(first));
  return rund::math32::detail::ScalarAddSat(
      rund::math32::detail::ScalarAddSat(ten, SmootherScale5_32(second)), t5);
}

[[nodiscard]] inline rund::kernel::i64
Smootherstep64(const rund::kernel::i64 edge0, const rund::kernel::i64 edge1,
               const rund::kernel::i64 value) noexcept {
  const rund::kernel::i64 t = Unlerp64(edge0, edge1, value);
  const rund::kernel::i64 inv = rund::math64::detail::ScalarSubSat(Max64(), t);
  const rund::kernel::i64 t2 = QuantizeProduct(t, t);
  const rund::kernel::i64 t3 = QuantizeProduct(t2, t);
  const rund::kernel::i64 t4 = QuantizeProduct(t3, t);
  const rund::kernel::i64 t5 = QuantizeProduct(t4, t);
  const rund::kernel::i64 inv2 = QuantizeProduct(inv, inv);
  const rund::kernel::i64 first = QuantizeProduct(t3, inv2);
  const rund::kernel::i64 second = QuantizeProduct(t4, inv);
  const rund::kernel::i64 ten =
      rund::math64::detail::ScalarAddSat(SmootherScale5_64(first),
                                         SmootherScale5_64(first));
  return rund::math64::detail::ScalarAddSat(
      rund::math64::detail::ScalarAddSat(ten, SmootherScale5_64(second)), t5);
}

template <typename T>
[[nodiscard]] inline T XorBits(const T lhs, const T rhs) noexcept {
  using U = std::conditional_t<sizeof(T) == sizeof(rund::kernel::i64),
                               rund::kernel::u64, rund::kernel::u32>;
  return std::bit_cast<T>(std::bit_cast<U>(lhs) ^ std::bit_cast<U>(rhs));
}

template <typename T>
[[nodiscard]] inline T AddWrap(const T lhs, const T rhs) noexcept {
  if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
    return rund::math64::detail::ScalarAddWrap(lhs, rhs);
  } else {
    return rund::math32::detail::ScalarAddWrap(lhs, rhs);
  }
}

namespace detail {

using U128 = __uint128_t;
using I128 = __int128_t;

template <typename T>
[[nodiscard]] constexpr U128 Magnitude(const T value) noexcept {
  using U = std::make_unsigned_t<T>;
  const U bits = static_cast<U>(value);
  return value < 0 ? static_cast<U128>(static_cast<U>(~bits)) + 1u
                   : static_cast<U128>(bits);
}

[[nodiscard]] constexpr U128 RoundShiftNearestEven(
    const U128 magnitude, const unsigned fraction) noexcept {
  if (fraction == 0u) {
    return magnitude;
  }
  U128 quotient = magnitude >> fraction;
  const U128 remainder =
      magnitude & ((U128{1u} << fraction) - 1u);
  const U128 half = U128{1u} << (fraction - 1u);
  if (remainder > half ||
      (remainder == half && (quotient & 1u) != 0u)) {
    ++quotient;
  }
  return quotient;
}

[[nodiscard]] constexpr U128 RoundDivideNearestEven(
    const U128 numerator, const U128 denominator) noexcept {
  U128 quotient = numerator / denominator;
  const U128 remainder = numerator % denominator;
  const U128 complement = denominator - remainder;
  if (remainder > complement ||
      (remainder == complement && (quotient & 1u) != 0u)) {
    ++quotient;
  }
  return quotient;
}

template <typename T>
[[nodiscard]] constexpr T SaturateSignedMagnitude(
    const U128 magnitude, const bool negative) noexcept {
  const U128 positive_max =
      static_cast<U128>(std::numeric_limits<T>::max());
  const U128 negative_max = positive_max + 1u;
  if (negative) {
    return magnitude >= negative_max
               ? std::numeric_limits<T>::min()
               : static_cast<T>(-static_cast<I128>(magnitude));
  }
  return magnitude > positive_max ? std::numeric_limits<T>::max()
                                  : static_cast<T>(magnitude);
}

template <typename T>
[[nodiscard]] constexpr T QuantizeSignedWide(
    const I128 value, const unsigned fraction) noexcept {
  const U128 bits = static_cast<U128>(value);
  const bool negative = value < 0;
  const U128 magnitude = negative ? (~bits + 1u) : bits;
  return SaturateSignedMagnitude<T>(
      RoundShiftNearestEven(magnitude, fraction), negative);
}

} // namespace detail

template <typename T>
[[nodiscard]] constexpr T DivNearestEven(
    const T lhs, const T rhs, const unsigned fraction) noexcept {
  if (rhs == 0) {
    return lhs == 0 ? T{0}
                    : lhs < 0 ? std::numeric_limits<T>::min()
                              : std::numeric_limits<T>::max();
  }
  const bool negative = (lhs < 0) != (rhs < 0);
  const detail::U128 numerator = detail::Magnitude(lhs) << fraction;
  const detail::U128 denominator = detail::Magnitude(rhs);
  return detail::SaturateSignedMagnitude<T>(
      detail::RoundDivideNearestEven(numerator, denominator), negative);
}

template <typename T, typename... Values>
[[nodiscard]] constexpr T QuantizeSum(const Values... values) noexcept {
  const rund::kernel::i128 sum =
      (static_cast<rund::kernel::i128>(values) + ... +
       rund::kernel::i128{0});
  const rund::kernel::i128 lower = std::numeric_limits<T>::min();
  const rund::kernel::i128 upper = std::numeric_limits<T>::max();
  return static_cast<T>(sum < lower ? lower : sum > upper ? upper : sum);
}

template <typename T>
[[nodiscard]] constexpr T QuantizeMulAdd(const T lhs, const T rhs,
                                         const T addend,
                                         const unsigned fraction =
                                             sizeof(T) * 8u - 1u) noexcept {
  const detail::I128 unit = detail::I128{1} << fraction;
  const detail::I128 value =
      static_cast<detail::I128>(lhs) * static_cast<detail::I128>(rhs) +
      static_cast<detail::I128>(addend) * unit;
  return detail::QuantizeSignedWide<T>(value, fraction);
}

template <typename T>
[[nodiscard]] constexpr T QuantizeProduct(
    const T lhs, const T rhs, const unsigned fraction) noexcept {
  const bool negative = (lhs < 0) != (rhs < 0);
  const detail::U128 magnitude =
      detail::Magnitude(lhs) * detail::Magnitude(rhs);
  return detail::SaturateSignedMagnitude<T>(
      detail::RoundShiftNearestEven(magnitude, fraction), negative);
}

template <typename T>
[[nodiscard]] constexpr T QuantizeScaledProduct(
    const T value, const std::make_unsigned_t<T> coefficient,
    const unsigned fraction = sizeof(T) * 8u - 1u) noexcept {
  const detail::U128 magnitude =
      detail::Magnitude(value) * static_cast<detail::U128>(coefficient);
  return detail::SaturateSignedMagnitude<T>(
      detail::RoundShiftNearestEven(magnitude, fraction), value < 0);
}

template <typename U>
[[nodiscard]] constexpr U QuantizeUnsignedProduct(
    const U lhs, const U rhs,
    const unsigned fraction = sizeof(U) * 8u - 1u) noexcept {
  static_assert(std::is_unsigned_v<U>);
  const detail::U128 magnitude =
      static_cast<detail::U128>(lhs) * static_cast<detail::U128>(rhs);
  const detail::U128 rounded =
      detail::RoundShiftNearestEven(magnitude, fraction);
  const detail::U128 maximum =
      static_cast<detail::U128>(std::numeric_limits<U>::max());
  return rounded > maximum ? std::numeric_limits<U>::max()
                           : static_cast<U>(rounded);
}

template <typename T>
[[nodiscard]] constexpr T RecipNearestEven(
    const T value,
    const unsigned fraction = sizeof(T) * 8u - 1u) noexcept {
  if (value == 0) {
    return std::numeric_limits<T>::max();
  }
  const detail::U128 unit = detail::U128{1u} << fraction;
  const detail::U128 quotient =
      detail::RoundDivideNearestEven(unit << fraction,
                                     detail::Magnitude(value));
  return detail::SaturateSignedMagnitude<T>(quotient, value < 0);
}

template <typename T>
[[nodiscard]] constexpr T SqrtFloor(
    const T value,
    const unsigned fraction = sizeof(T) * 8u - 1u) noexcept {
  if (value <= 0) {
    return 0;
  }
  const detail::U128 radicand =
      static_cast<detail::U128>(static_cast<std::make_unsigned_t<T>>(value))
      << fraction;
  detail::U128 root = 0u;
  for (unsigned step = 0u; step < 64u; ++step) {
    const detail::U128 candidate =
        root | (detail::U128{1u} << (63u - step));
    if (candidate * candidate <= radicand) {
      root = candidate;
    }
  }
  const detail::U128 maximum =
      static_cast<detail::U128>(std::numeric_limits<T>::max());
  return root > maximum ? std::numeric_limits<T>::max()
                        : static_cast<T>(root);
}

template <typename T>
[[nodiscard]] constexpr T RsqrtNearestEven(
    const T value,
    const unsigned fraction = sizeof(T) * 8u - 1u) noexcept {
  return RecipNearestEven(SqrtFloor(value, fraction), fraction);
}

} // namespace node_accel_contract::cpu::fixed
