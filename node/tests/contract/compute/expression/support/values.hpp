#pragma once

#include <rund/compute/fixed.hpp>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace rund::node::test_contract::expression {
namespace {

template <std::size_t> struct Field final {};

template <class T> inline constexpr bool is_fixed = false;
template <unsigned I, unsigned F>
inline constexpr bool is_fixed<rund::compute::Fixed<I, F>> = true;

template <class T> struct alternate_fixed;
template <> struct alternate_fixed<rund::compute::Fixed<16, 16>> final {
  using Type = rund::compute::Fixed<17, 15>;
};
template <> struct alternate_fixed<rund::compute::Fixed<20, 44>> final {
  using Type = rund::compute::Fixed<21, 43>;
};

template <class T>
[[nodiscard]] constexpr T fixed_value(const int numerator,
                                      const int denominator) {
  return T::from_raw(static_cast<typename T::Raw>(
      (static_cast<__int128_t>(numerator) << T::fraction_bits) / denominator));
}

template <class T>
[[nodiscard]] constexpr T fixed_nearest(const std::uint64_t numerator,
                                        const std::uint64_t denominator) {
  using Raw = typename T::Raw;
  __uint128_t scaled = static_cast<__uint128_t>(numerator) << T::fraction_bits;
  __uint128_t quotient = scaled / denominator;
  const __uint128_t remainder = scaled % denominator;
  const __uint128_t twice = remainder << 1u;
  if (twice > denominator || (twice == denominator && (quotient & 1u) != 0u)) {
    ++quotient;
  }
  const auto maximum =
      static_cast<__uint128_t>(std::numeric_limits<Raw>::max());
  return T::from_raw(static_cast<Raw>(quotient > maximum ? maximum : quotient));
}

template <class T>
[[nodiscard]] constexpr T fixed_q31(const std::uint32_t bits) {
  return fixed_nearest<T>(bits, std::uint64_t{1} << 31u);
}

template <class T>
[[nodiscard]] constexpr T fixed_unit_hash(const T value,
                                          const T seed = T::zero()) {
  using Raw = typename T::Raw;
  using Unsigned = std::make_unsigned_t<Raw>;
  Unsigned bits = std::bit_cast<Unsigned>(value.raw()) ^
                  std::bit_cast<Unsigned>(seed.raw());
  if constexpr (sizeof(Raw) == 8u) {
    bits ^= bits >> 33u;
    bits *= static_cast<Unsigned>(0xff51afd7ed558ccdull);
    bits ^= bits >> 33u;
    bits *= static_cast<Unsigned>(0xc4ceb9fe1a85ec53ull);
    bits ^= bits >> 33u;
  } else {
    bits ^= bits >> 16u;
    bits *= static_cast<Unsigned>(0x7feb352dull);
    bits ^= bits >> 15u;
    bits *= static_cast<Unsigned>(0x846ca68bull);
    bits ^= bits >> 16u;
  }
  constexpr Unsigned mask = (Unsigned{1} << T::fraction_bits) - Unsigned{1};
  return T::from_raw(std::bit_cast<Raw>(bits & mask));
}

template <class T> [[nodiscard]] long long printable(const T value) {
  if constexpr (is_fixed<T>) {
    return static_cast<long long>(value.raw());
  } else {
    return static_cast<long long>(value);
  }
}

template <class T> [[nodiscard]] constexpr std::array<T, 3u> helper_input() {
  return {fixed_value<T>(-1, 4), fixed_value<T>(1, 4), fixed_value<T>(1, 2)};
}

} // namespace
} // namespace rund::node::test_contract::expression
