#pragma once

#include <rund/compute/math.hpp>

#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace rund::node::test_contract::declared_detail {

template <std::size_t> struct DeclaredMultiplySlot final {};

template <class T>
using FixedUnsignedRaw = std::make_unsigned_t<typename T::Raw>;

template <class T>
[[nodiscard]] constexpr FixedUnsignedRaw<T> FixedRawBits(const T value) {
  return std::bit_cast<FixedUnsignedRaw<T>>(value.raw());
}

[[nodiscard]] constexpr __int128_t
RoundSignedFixed(const __int128_t value, const unsigned shift,
                 const rund::compute::Rounding rounding) {
  using UWide = __uint128_t;
  const bool negative = value < 0;
  const UWide magnitude = negative ? static_cast<UWide>(-(value + 1)) + 1u
                                   : static_cast<UWide>(value);
  UWide quotient = magnitude >> shift;
  const UWide remainder = magnitude & ((static_cast<UWide>(1u) << shift) - 1u);
  const UWide half = static_cast<UWide>(1u) << (shift - 1u);
  const bool nonzero = remainder != 0u;
  const bool nearest =
      remainder > half || (remainder == half && (quotient & 1u) != 0u);
  if ((rounding == rund::compute::Rounding::Down && negative && nonzero) ||
      (rounding == rund::compute::Rounding::Up && !negative && nonzero) ||
      (rounding == rund::compute::Rounding::NearestEven && nearest)) {
    ++quotient;
  }
  return negative ? -static_cast<__int128_t>(quotient)
                  : static_cast<__int128_t>(quotient);
}

[[nodiscard]] constexpr __uint128_t
RoundUnsignedFixed(const __uint128_t value, const unsigned shift,
                   const rund::compute::Rounding rounding) {
  __uint128_t quotient = value >> shift;
  const __uint128_t remainder =
      value & ((static_cast<__uint128_t>(1u) << shift) - 1u);
  const __uint128_t half = static_cast<__uint128_t>(1u) << (shift - 1u);
  const bool nonzero = remainder != 0u;
  const bool nearest =
      remainder > half || (remainder == half && (quotient & 1u) != 0u);
  if ((rounding == rund::compute::Rounding::Up && nonzero) ||
      (rounding == rund::compute::Rounding::NearestEven && nearest)) {
    ++quotient;
  }
  return quotient;
}

template <class T>
[[nodiscard]] constexpr T
NarrowSignedFixed(const __int128_t value,
                  const rund::compute::Overflow overflow) {
  using Raw = typename T::Raw;
  using Unsigned = FixedUnsignedRaw<T>;
  if (overflow == rund::compute::Overflow::Saturate) {
    constexpr auto low =
        static_cast<__int128_t>(std::numeric_limits<Raw>::min());
    constexpr auto high =
        static_cast<__int128_t>(std::numeric_limits<Raw>::max());
    return T::from_raw(value < low    ? std::numeric_limits<Raw>::min()
                       : value > high ? std::numeric_limits<Raw>::max()
                                      : static_cast<Raw>(value));
  }
  constexpr __uint128_t mask =
      static_cast<__uint128_t>(std::numeric_limits<Unsigned>::max());
  return T::from_raw(std::bit_cast<Raw>(
      static_cast<Unsigned>(static_cast<__uint128_t>(value) & mask)));
}

template <class T>
[[nodiscard]] constexpr T
NarrowUnsignedFixed(const __uint128_t value,
                    const rund::compute::Overflow overflow) {
  using Raw = typename T::Raw;
  using Unsigned = FixedUnsignedRaw<T>;
  constexpr auto maximum =
      static_cast<__uint128_t>(std::numeric_limits<Unsigned>::max());
  const auto narrowed = static_cast<Unsigned>(
      overflow == rund::compute::Overflow::Saturate && value > maximum
          ? maximum
          : value & maximum);
  return T::from_raw(std::bit_cast<Raw>(narrowed));
}

template <class T>
[[nodiscard]] constexpr T
ReferenceSignedMultiply(const T left, const T right,
                        const rund::compute::Rounding rounding,
                        const rund::compute::Overflow overflow) {
  const auto product = static_cast<__int128_t>(left.raw()) *
                       static_cast<__int128_t>(right.raw());
  return NarrowSignedFixed<T>(
      RoundSignedFixed(product, T::fraction_bits, rounding), overflow);
}

template <class T>
[[nodiscard]] constexpr T
ReferenceScaledMultiply(const T left, const T right,
                        const rund::compute::Rounding rounding,
                        const rund::compute::Overflow overflow) {
  const auto product = static_cast<__int128_t>(left.raw()) *
                       static_cast<__int128_t>(FixedRawBits(right));
  return NarrowSignedFixed<T>(
      RoundSignedFixed(product, T::fraction_bits, rounding), overflow);
}

template <class T>
[[nodiscard]] constexpr T
ReferenceUnsignedMultiply(const T left, const T right,
                          const rund::compute::Rounding rounding,
                          const rund::compute::Overflow overflow) {
  const auto product = static_cast<__uint128_t>(FixedRawBits(left)) *
                       static_cast<__uint128_t>(FixedRawBits(right));
  return NarrowUnsignedFixed<T>(
      RoundUnsignedFixed(product, T::fraction_bits, rounding), overflow);
}

template <class T>
[[nodiscard]] constexpr T
ReferenceMulAdd(const T left, const T right, const T addend,
                const rund::compute::Rounding rounding,
                const rund::compute::Overflow overflow) {
  const __int128_t scale = static_cast<__int128_t>(1u) << T::fraction_bits;
  const __int128_t widened = static_cast<__int128_t>(left.raw()) *
                                 static_cast<__int128_t>(right.raw()) +
                             static_cast<__int128_t>(addend.raw()) * scale;
  return NarrowSignedFixed<T>(
      RoundSignedFixed(widened, T::fraction_bits, rounding), overflow);
}

} // namespace rund::node::test_contract::declared_detail
