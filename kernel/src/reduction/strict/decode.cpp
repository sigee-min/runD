#include "local.hpp"

namespace rund::kernel::strict_float {

Decoded Decode(const u64 bits, const Format& format) {
  const u64 raw_exponent = (bits & format.exponent_mask) >> format.fraction_bits;
  const u64 fraction = bits & format.fraction_mask;
  Decoded out{
      .sign = (bits & format.sign_mask) != 0u,
  };
  if (raw_exponent == format.exponent_max) {
    out.inf = fraction == 0u;
    out.nan = fraction != 0u;
    return out;
  }
  if (raw_exponent == 0u) {
    out.zero = fraction == 0u;
    out.exponent = 1 - format.exponent_bias;
    out.mantissa = fraction;
    return out;
  }
  out.exponent = static_cast<i32>(raw_exponent) - format.exponent_bias;
  out.mantissa = (static_cast<u128>(1u) << format.fraction_bits) | fraction;
  return out;
}

} // namespace rund::kernel::strict_float
