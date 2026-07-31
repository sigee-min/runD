#pragma once

#include "float.hpp"

namespace rund::kernel::strict_float {

struct Format {
  u32 exponent_bits = 0u;
  u32 fraction_bits = 0u;
  u64 sign_mask = 0u;
  u64 exponent_mask = 0u;
  u64 fraction_mask = 0u;
  u32 exponent_max = 0u;
  i32 exponent_bias = 0;
  u64 canonical_nan = 0u;
};

struct Decoded {
  bool sign = false;
  bool zero = false;
  bool inf = false;
  bool nan = false;
  i32 exponent = 0;
  u128 mantissa = 0u;
};

Format Float32Format();
Format Float64Format();
Decoded Decode(u64 bits, const Format& format);
u128 ShiftRightJam128(u128 value, u32 shift);
u64 Compose(bool sign, u32 raw_exponent, u64 fraction, const Format& format);
u64 Infinity(bool sign, const Format& format);
u64 Zero(bool sign, const Format& format);
u64 AddBits(u64 left_bits,
            u64 right_bits,
            const Format& format,
            StrictFloatReductionPolicy policy);

} // namespace rund::kernel::strict_float
