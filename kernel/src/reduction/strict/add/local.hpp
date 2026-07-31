#pragma once

#include "../local.hpp"

namespace rund::kernel::strict_float {

inline constexpr u32 kAddGuardBits = 3u;

struct FiniteSum {
  bool zero = false;
  bool sign = false;
  i32 exponent = 0;
  u128 mantissa = 0u;
};

bool ResolveSpecialAddResult(const Decoded& left,
                             const Decoded& right,
                             u64 left_bits,
                             u64 right_bits,
                             const Format& format,
                             StrictFloatReductionPolicy policy,
                             u64& out_result);
FiniteSum AddFiniteMantissas(const Decoded& left, const Decoded& right);
u64 ComposeRoundedFiniteSum(FiniteSum sum, const Format& format);

} // namespace rund::kernel::strict_float
