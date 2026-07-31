#include "local.hpp"

namespace rund::kernel::strict_float {
namespace {

u64 ComposeRoundedNormal(bool result_sign,
                         i32 common_exponent,
                         u128 mantissa,
                         const Format& format) {
  const u128 round_bits =
      mantissa & ((static_cast<u128>(1u) << kAddGuardBits) - 1u);
  u128 rounded = mantissa >> kAddGuardBits;
  const u128 half = static_cast<u128>(1u) << (kAddGuardBits - 1u);
  if (round_bits > half || (round_bits == half && ((rounded & 1u) != 0u))) {
    rounded += 1u;
  }
  if (rounded >= (static_cast<u128>(1u) << (format.fraction_bits + 1u))) {
    rounded >>= 1u;
    common_exponent += 1;
  }
  const i32 raw_exponent = common_exponent + format.exponent_bias;
  if (raw_exponent >= static_cast<i32>(format.exponent_max)) {
    return Infinity(result_sign, format);
  }
  const u128 hidden = static_cast<u128>(1u) << format.fraction_bits;
  if (raw_exponent <= 0 || rounded < hidden) {
    return Compose(result_sign, 0u, static_cast<u64>(rounded), format);
  }
  return Compose(result_sign,
                 static_cast<u32>(raw_exponent),
                 static_cast<u64>(rounded - hidden),
                 format);
}

} // namespace

u64 ComposeRoundedFiniteSum(FiniteSum sum, const Format& format) {
  if (sum.zero) {
    return Zero(false, format);
  }
  const u128 normalized_min =
      (static_cast<u128>(1u) << format.fraction_bits) << kAddGuardBits;
  const u128 normalized_max = normalized_min << 1u;
  while (sum.mantissa >= normalized_max) {
    sum.mantissa = ShiftRightJam128(sum.mantissa, 1u);
    sum.exponent += 1;
  }
  while (sum.mantissa < normalized_min && sum.exponent > (1 - format.exponent_bias)) {
    sum.mantissa <<= 1u;
    sum.exponent -= 1;
  }
  const i32 min_exponent = 1 - format.exponent_bias;
  if (sum.exponent < min_exponent) {
    sum.mantissa = ShiftRightJam128(sum.mantissa, static_cast<u32>(min_exponent - sum.exponent));
    sum.exponent = min_exponent;
  }
  return ComposeRoundedNormal(sum.sign, sum.exponent, sum.mantissa, format);
}

} // namespace rund::kernel::strict_float
