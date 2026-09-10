#pragma once

[[nodiscard]] inline WideScalar
SignedWideFromBits(const __uint128_t bits) noexcept {
  return std::bit_cast<WideScalar>(bits);
}

[[nodiscard]] inline WideScalar ShiftWideLeft(const WideScalar value,
                                              const unsigned shift) noexcept {
  return SignedWideFromBits(static_cast<__uint128_t>(value) << shift);
}

[[nodiscard]] inline WideScalar
AlignWideFraction(const WideScalar value, const unsigned source_fraction,
                  const unsigned target_fraction) noexcept {
  return source_fraction == target_fraction
             ? value
             : ShiftWideLeft(value, target_fraction - source_fraction);
}

[[nodiscard]] inline WideScalar
QuantizeWide(const WideScalar value,
             const rund::kernel::ComputeFixedFormat source,
             const rund::kernel::ComputeFixedFormat target) noexcept {
  const unsigned width =
      static_cast<unsigned>(target.integer_bits) + target.fraction_bits;
  const __uint128_t sign = static_cast<__uint128_t>(1u) << (width - 1u);
  const WideScalar low = -static_cast<WideScalar>(sign);
  const WideScalar high = static_cast<WideScalar>(sign - 1u);
  WideScalar scaled = value;
  if (target.fraction_bits > source.fraction_bits) {
    const unsigned shift =
        static_cast<unsigned>(target.fraction_bits - source.fraction_bits);
    if (target.overflow == rund::kernel::ComputeOverflow::Saturate) {
      const WideScalar factor = static_cast<WideScalar>(1u) << shift;
      const WideScalar low_before_scale = low / factor;
      const WideScalar high_before_scale = high / factor;
      if (value < low_before_scale) {
        return low;
      }
      if (value > high_before_scale) {
        return high;
      }
    }
    scaled = ShiftWideLeft(scaled, shift);
  } else if (target.fraction_bits < source.fraction_bits) {
    const unsigned shift =
        static_cast<unsigned>(source.fraction_bits - target.fraction_bits);
    const bool negative = scaled < 0;
    const __uint128_t bits = static_cast<__uint128_t>(scaled);
    const __uint128_t magnitude = negative ? (~bits + 1u) : bits;
    __uint128_t quotient = magnitude >> shift;
    const __uint128_t mask = (static_cast<__uint128_t>(1u) << shift) - 1u;
    const __uint128_t remainder = magnitude & mask;
    const __uint128_t halfway = static_cast<__uint128_t>(1u) << (shift - 1u);
    const bool nonzero = remainder != 0u;
    const bool nearest =
        remainder > halfway || (remainder == halfway && (quotient & 1u) != 0u);
    if ((target.rounding == rund::kernel::ComputeRounding::Down && negative &&
         nonzero) ||
        (target.rounding == rund::kernel::ComputeRounding::Up && !negative &&
         nonzero) ||
        (target.rounding == rund::kernel::ComputeRounding::NearestEven &&
         nearest)) {
      ++quotient;
    }
    scaled = negative ? SignedWideFromBits(~quotient + 1u)
                      : SignedWideFromBits(quotient);
  }

  if (target.overflow == rund::kernel::ComputeOverflow::Saturate) {
    return scaled < low ? low : scaled > high ? high : scaled;
  }
  const __uint128_t mask = width == 128u
                               ? ~static_cast<__uint128_t>(0u)
                               : (static_cast<__uint128_t>(1u) << width) - 1u;
  __uint128_t wrapped = static_cast<__uint128_t>(scaled) & mask;
  if (width < 128u && (wrapped & sign) != 0u) {
    wrapped |= ~mask;
  }
  return SignedWideFromBits(wrapped);
}
