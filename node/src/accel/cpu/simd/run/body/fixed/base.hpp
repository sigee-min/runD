#pragma once

namespace rund::node::accel::cpu_simd_detail {
namespace {

using UnsignedWideScalar = __uint128_t;

[[nodiscard]] inline UnsignedWideScalar
WideMagnitude(const WideScalar value) noexcept {
  return value < 0 ? static_cast<UnsignedWideScalar>(-(value + 1)) + 1u
                   : static_cast<UnsignedWideScalar>(value);
}

[[nodiscard]] inline unsigned
StoredWidth(const rund::kernel::ComputeFixedFormat format) noexcept {
  return static_cast<unsigned>(format.integer_bits) + format.fraction_bits;
}

[[nodiscard]] inline UnsignedWideScalar
StoredUnsignedBits(const WideScalar value,
                   const rund::kernel::ComputeFixedFormat format) noexcept {
  const unsigned width = StoredWidth(format);
  const UnsignedWideScalar mask = width == 128u
                                      ? ~UnsignedWideScalar{0u}
                                      : (UnsignedWideScalar{1u} << width) - 1u;
  return static_cast<UnsignedWideScalar>(value) & mask;
}

[[nodiscard]] inline WideScalar
StoredSignedBits(UnsignedWideScalar value,
                 const rund::kernel::ComputeFixedFormat format) noexcept {
  const unsigned width = StoredWidth(format);
  const UnsignedWideScalar sign = UnsignedWideScalar{1u} << (width - 1u);
  const UnsignedWideScalar mask = width == 128u
                                      ? ~UnsignedWideScalar{0u}
                                      : (UnsignedWideScalar{1u} << width) - 1u;
  value &= mask;
  if (width < 128u && (value & sign) != 0u) {
    value |= ~mask;
  }
  return SignedWideFromBits(value);
}

[[nodiscard]] inline rund::kernel::ComputeFixedFormat
FixedProductFormat(const rund::kernel::ComputeFixedFormat format) noexcept {
  auto product = format;
  product.integer_bits = static_cast<rund::kernel::u8>(
      static_cast<unsigned>(format.integer_bits) * 2u);
  product.fraction_bits = static_cast<rund::kernel::u8>(
      static_cast<unsigned>(format.fraction_bits) * 2u);
  return product;
}

[[nodiscard]] inline WideScalar QuantizeUnsignedFixedProduct(
    const UnsignedWideScalar product,
    const rund::kernel::ComputeFixedFormat format) noexcept {
  const unsigned shift = format.fraction_bits;
  UnsignedWideScalar quotient = product >> shift;
  const UnsignedWideScalar mask = (UnsignedWideScalar{1u} << shift) - 1u;
  const UnsignedWideScalar remainder = product & mask;
  const UnsignedWideScalar halfway = UnsignedWideScalar{1u} << (shift - 1u);
  const bool nonzero = remainder != 0u;
  const bool nearest =
      remainder > halfway || (remainder == halfway && (quotient & 1u) != 0u);
  if ((format.rounding == rund::kernel::ComputeRounding::Up && nonzero) ||
      (format.rounding == rund::kernel::ComputeRounding::NearestEven &&
       nearest)) {
    ++quotient;
  }
  const unsigned width = StoredWidth(format);
  const UnsignedWideScalar maximum = (UnsignedWideScalar{1u} << width) - 1u;
  if (format.overflow == rund::kernel::ComputeOverflow::Saturate &&
      quotient > maximum) {
    quotient = maximum;
  }
  return StoredSignedBits(quotient & maximum, format);
}

[[nodiscard]] inline WideScalar
FixedDivideRaw(const WideScalar lhs, const WideScalar rhs,
               const rund::kernel::ComputeFixedFormat format) noexcept {
  const unsigned fraction = format.fraction_bits;
  if (rhs == 0) {
    if (lhs == 0) {
      return 0;
    }
    const UnsignedWideScalar sign = UnsignedWideScalar{1u}
                                    << (format.integer_bits + fraction - 1u);
    return lhs < 0 ? -static_cast<WideScalar>(sign)
                   : static_cast<WideScalar>(sign - 1u);
  }
  const bool negative = (lhs < 0) != (rhs < 0);
  const UnsignedWideScalar numerator = WideMagnitude(lhs) << fraction;
  const UnsignedWideScalar denominator = WideMagnitude(rhs);
  UnsignedWideScalar quotient = numerator / denominator;
  const UnsignedWideScalar remainder = numerator % denominator;
  const bool nonzero = remainder != 0u;
  const UnsignedWideScalar twice = remainder << 1u;
  const bool nearest =
      twice > denominator || (twice == denominator && (quotient & 1u) != 0u);
  if ((format.rounding == rund::kernel::ComputeRounding::Down && negative &&
       nonzero) ||
      (format.rounding == rund::kernel::ComputeRounding::Up && !negative &&
       nonzero) ||
      (format.rounding == rund::kernel::ComputeRounding::NearestEven &&
       nearest)) {
    ++quotient;
  }
  return negative ? -static_cast<WideScalar>(quotient)
                  : static_cast<WideScalar>(quotient);
}

[[nodiscard]] inline WideScalar
FixedSqrtRaw(const WideScalar value,
             const rund::kernel::ComputeFixedFormat format) noexcept {
  if (value <= 0) {
    return 0;
  }
  const UnsignedWideScalar radicand = static_cast<UnsignedWideScalar>(value)
                                      << format.fraction_bits;
  UnsignedWideScalar root = 0u;
  for (unsigned step = 0u; step < 64u; ++step) {
    const UnsignedWideScalar candidate =
        root | (UnsignedWideScalar{1u} << (63u - step));
    if (candidate * candidate <= radicand) {
      root = candidate;
    }
  }
  return static_cast<WideScalar>(root);
}

} // namespace
} // namespace rund::node::accel::cpu_simd_detail
