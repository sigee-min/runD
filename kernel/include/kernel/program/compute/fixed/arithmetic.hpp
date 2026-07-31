#pragma once

#include <kernel/program/compute/model.hpp>

#include <bit>
#include <limits>
#include <type_traits>

namespace rund::kernel::compute_fixed_detail {

template <typename S>
[[nodiscard]] constexpr S Narrow(const i128 value,
                                 const ComputeOverflow overflow) noexcept {
  if (overflow == ComputeOverflow::Saturate) {
    const i128 low = static_cast<i128>(std::numeric_limits<S>::min());
    const i128 high = static_cast<i128>(std::numeric_limits<S>::max());
    return value < low ? std::numeric_limits<S>::min()
                       : value > high ? std::numeric_limits<S>::max()
                                      : static_cast<S>(value);
  }
  using U = std::make_unsigned_t<S>;
  return std::bit_cast<S>(static_cast<U>(static_cast<u128>(value)));
}

[[nodiscard]] constexpr i128 RoundQuotient(
    i128 quotient, const i128 remainder, const i128 divisor,
    const bool negative, const ComputeRounding rounding) noexcept {
  if (remainder == 0) { return quotient; }
  if (rounding == ComputeRounding::Down && negative) { return quotient - 1; }
  if (rounding == ComputeRounding::Up && !negative) { return quotient + 1; }
  if (rounding != ComputeRounding::NearestEven) { return quotient; }
  const u128 magnitude =
      remainder < 0 ? static_cast<u128>(-remainder)
                    : static_cast<u128>(remainder);
  const u128 scale =
      divisor < 0 ? static_cast<u128>(-divisor)
                  : static_cast<u128>(divisor);
  const u128 twice = magnitude * 2u;
  if (twice > scale || (twice == scale && (quotient & 1) != 0)) {
    return quotient + (negative ? -1 : 1);
  }
  return quotient;
}

template <typename S>
[[nodiscard]] constexpr S Mul(const S lhs, const S rhs,
                              const ComputeFixedFormat format) noexcept {
  const i128 scale = static_cast<i128>(1u) << format.fraction_bits;
  const i128 product = static_cast<i128>(lhs) * rhs;
  i128 quotient = product / scale;
  quotient = RoundQuotient(quotient, product % scale, scale, product < 0,
                           format.rounding);
  return Narrow<S>(quotient, format.overflow);
}

template <typename S, typename Source>
[[nodiscard]] constexpr S Rescale(const Source value,
                                  const u32 source_fraction_bits,
                                  const ComputeFixedFormat format) noexcept {
  i128 scaled = static_cast<i128>(value);
  if (source_fraction_bits > format.fraction_bits) {
    const u32 shift = source_fraction_bits - format.fraction_bits;
    const i128 divisor = static_cast<i128>(1u) << shift;
    i128 quotient = scaled / divisor;
    quotient = RoundQuotient(quotient, scaled % divisor, divisor, scaled < 0,
                             format.rounding);
    scaled = quotient;
  } else if (source_fraction_bits < format.fraction_bits) {
    const u32 shift = format.fraction_bits - source_fraction_bits;
    const i128 factor = static_cast<i128>(1u) << shift;
    if (format.overflow == ComputeOverflow::Saturate) {
      const i128 low =
          static_cast<i128>(std::numeric_limits<S>::min());
      const i128 high =
          static_cast<i128>(std::numeric_limits<S>::max());
      if (scaled < low / factor) {
        return std::numeric_limits<S>::min();
      }
      if (scaled > high / factor) {
        return std::numeric_limits<S>::max();
      }
      scaled *= factor;
    } else {
      scaled = std::bit_cast<i128>(
          static_cast<u128>(scaled) << shift);
    }
  }
  return Narrow<S>(scaled, format.overflow);
}

template <typename S>
[[nodiscard]] constexpr S Div(const S lhs, const S rhs,
                              const ComputeFixedFormat format) noexcept {
  if (rhs == 0) { return S{0}; }
  const i128 numerator =
      static_cast<i128>(lhs) *
      (static_cast<i128>(1u) << format.fraction_bits);
  i128 quotient = numerator / rhs;
  quotient = RoundQuotient(quotient, numerator % rhs, rhs,
                           (lhs < 0) != (rhs < 0), format.rounding);
  return Narrow<S>(quotient, format.overflow);
}

template <typename S>
[[nodiscard]] constexpr S Sqrt(const S value,
                               const ComputeFixedFormat format) noexcept {
  if (value <= 0) { return S{0}; }
  const u128 target =
      static_cast<u128>(value) << format.fraction_bits;
  u128 root = 0u;
  u128 bit = static_cast<u128>(1u) << 126u;
  while (bit > target) { bit >>= 2u; }
  u128 remainder = target;
  while (bit != 0u) {
    if (remainder >= root + bit) {
      remainder -= root + bit;
      root = (root >> 1u) + bit;
    } else {
      root >>= 1u;
    }
    bit >>= 2u;
  }
  if (remainder != 0u) {
    if (format.rounding == ComputeRounding::Up) {
      ++root;
    } else if (format.rounding == ComputeRounding::NearestEven) {
      const u128 upper_distance = root * 2u + 1u - remainder;
      if (remainder > upper_distance ||
          (remainder == upper_distance && (root & 1u) != 0u)) {
        ++root;
      }
    }
  }
  return Narrow<S>(static_cast<i128>(root), format.overflow);
}

} // namespace rund::kernel::compute_fixed_detail
