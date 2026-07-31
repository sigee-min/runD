#pragma once

namespace rund::compute_dsl {

namespace detail {

[[nodiscard]] inline rund::kernel::ComputeFixedFormat FixedLiteralFormat(
    const ComputeValue anchor) noexcept {
  rund::kernel::ComputeFixedFormat format = FixedFormatOf(anchor);
  if (!rund::kernel::ComputeFixedFormatAbsent(format)) {
    format.approximation = rund::kernel::ComputeApproximation::Exact;
  }
  return format;
}

[[nodiscard]] inline rund::kernel::u64 FixedLaneMaximumBits(
    const ComputeValue anchor) noexcept {
  return ScalarModeOf(anchor) == ScalarMode::FixedLane64
             ? 0x7fffffffffffffffull
             : 0x7fffffffull;
}

[[nodiscard]] inline rund::kernel::u64 FixedLaneMinimumBits(
    const ComputeValue anchor) noexcept {
  return ScalarModeOf(anchor) == ScalarMode::FixedLane64
             ? 0x8000000000000000ull
             : 0x80000000ull;
}

[[nodiscard]] inline rund::kernel::u64 FixedNearestRatioBits(
    const ComputeValue anchor, const rund::kernel::u64 numerator,
    const rund::kernel::u64 denominator) noexcept {
  const rund::kernel::ComputeFixedFormat format = FixedFormatOf(anchor);
  if (denominator == 0u ||
      rund::kernel::ComputeFixedFormatAbsent(format)) {
    return 0u;
  }
  const __uint128_t scaled =
      static_cast<__uint128_t>(numerator) << format.fraction_bits;
  __uint128_t quotient = scaled / denominator;
  const __uint128_t remainder = scaled % denominator;
  const __uint128_t twice = remainder << 1u;
  if (twice > denominator ||
      (twice == denominator && (quotient & 1u) != 0u)) {
    ++quotient;
  }
  const rund::kernel::u64 maximum = FixedLaneMaximumBits(anchor);
  return quotient > maximum ? maximum
                            : static_cast<rund::kernel::u64>(quotient);
}

[[nodiscard]] inline ComputeValue FixedLiteral(
    const ComputeValue anchor, const rund::kernel::u64 bits) noexcept {
  const rund::kernel::ComputeFixedFormat format = FixedLiteralFormat(anchor);
  return rund::kernel::ComputeFixedFormatAbsent(format)
             ? Constant(anchor, bits)
             : FormattedConstant(anchor, bits, format);
}

[[nodiscard]] inline ComputeValue FixedRatioConstant(
    const ComputeValue anchor, const rund::kernel::u64 numerator,
    const rund::kernel::u64 denominator) noexcept {
  return FixedLiteral(
      anchor, FixedNearestRatioBits(anchor, numerator, denominator));
}

[[nodiscard]] inline ComputeValue FixedQ31Constant(
    const ComputeValue anchor, const rund::kernel::u32 bits) noexcept {
  return FixedRatioConstant(anchor, bits, rund::kernel::u64{1} << 31u);
}

[[nodiscard]] inline ComputeValue FixedFractionMaskConstant(
    const ComputeValue anchor) noexcept {
  const rund::kernel::ComputeFixedFormat format = FixedFormatOf(anchor);
  if (rund::kernel::ComputeFixedFormatAbsent(format)) {
    return FixedLiteral(anchor, FixedLaneMaximumBits(anchor));
  }
  const rund::kernel::u64 mask =
      (rund::kernel::u64{1} << format.fraction_bits) - 1u;
  return FixedLiteral(anchor, mask);
}

} // namespace detail

[[nodiscard]] inline ComputeValue
fixed_zero(const ComputeValue value) noexcept {
  return detail::FixedLiteral(value, 0u);
}

[[nodiscard]] inline ComputeValue fixed_max(const ComputeValue value) noexcept {
  return detail::FixedLiteral(value, detail::FixedLaneMaximumBits(value));
}

[[nodiscard]] inline ComputeValue fixed_one(const ComputeValue value) noexcept {
  return detail::FixedRatioConstant(value, 1u, 1u);
}

} // namespace rund::compute_dsl
