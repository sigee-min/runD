#pragma once

#include <string>

namespace rund::kernel::compute_lowering_detail {

inline void AppendMetalU128Core(std::string &out) {
  out += R"METAL(
struct RundU128 {
  ulong hi;
  ulong lo;
};
inline RundU128 RundMakeU128(ulong hi, ulong lo) {
  RundU128 value;
  value.hi = hi;
  value.lo = lo;
  return value;
}
inline bool RundGeU128(RundU128 lhs, RundU128 rhs) {
  return lhs.hi > rhs.hi || (lhs.hi == rhs.hi && lhs.lo >= rhs.lo);
}
inline RundU128 RundAddU128(RundU128 lhs, RundU128 rhs) {
  const ulong lo = lhs.lo + rhs.lo;
  return RundMakeU128(lhs.hi + rhs.hi + (lo < lhs.lo ? 1ul : 0ul), lo);
}
inline RundU128 RundSubU128(RundU128 lhs, RundU128 rhs) {
  const ulong borrow = lhs.lo < rhs.lo ? 1ul : 0ul;
  return RundMakeU128(lhs.hi - rhs.hi - borrow, lhs.lo - rhs.lo);
}
inline RundU128 RundShr1U128(RundU128 value) {
  return RundMakeU128(value.hi >> 1u,
                      (value.lo >> 1u) | (value.hi << 63u));
}
inline RundU128 RundShr2U128(RundU128 value) {
  return RundMakeU128(value.hi >> 2u,
                      (value.lo >> 2u) | (value.hi << 62u));
}
inline RundU128 RundShl1OrU128(RundU128 value, ulong bit) {
  return RundMakeU128((value.hi << 1u) | (value.lo >> 63u),
                      (value.lo << 1u) | bit);
}
inline ulong RundBitU128(RundU128 value, int bit) {
  return bit >= 64 ? ((value.hi >> uint(bit - 64)) & 1ul)
                   : ((value.lo >> uint(bit)) & 1ul);
}
inline RundU128 RundMulWide64(ulong lhs, ulong rhs) {
  const ulong mask = 0xfffffffful;
  const ulong lhs0 = lhs & mask;
  const ulong lhs1 = lhs >> 32u;
  const ulong rhs0 = rhs & mask;
  const ulong rhs1 = rhs >> 32u;
  const ulong product0 = lhs0 * rhs0;
  ulong middle = lhs1 * rhs0 + (product0 >> 32u);
  const ulong carry = middle >> 32u;
  middle = (middle & mask) + lhs0 * rhs1;
  const ulong high = lhs1 * rhs1 + carry + (middle >> 32u);
  const ulong low = (middle << 32u) | (product0 & mask);
  return RundMakeU128(high, low);
}
)METAL";
}

inline void AppendMetalU128Division(std::string &out) {
  out += R"METAL(
inline ulong RundUnsignedDivU128ByU64(RundU128 numerator, ulong denominator) {
  if (denominator == 0ul) { return 0xfffffffffffffffful; }
  ulong quotient = 0ul;
  RundU128 remainder = RundMakeU128(0ul, 0ul);
  bool overflow = false;
  for (int bit = 127; bit >= 0; --bit) {
    remainder = RundShl1OrU128(remainder, RundBitU128(numerator, bit));
    if (remainder.hi != 0ul || remainder.lo >= denominator) {
      if (remainder.lo < denominator) { remainder.hi -= 1ul; }
      remainder.lo -= denominator;
      if (bit >= 64) {
        overflow = true;
      } else {
        quotient |= 1ul << uint(bit);
      }
    }
  }
  return overflow ? 0xfffffffffffffffful : quotient;
}
)METAL";
}

inline void AppendMetalU128IntegerSquareRoot(std::string &out) {
  out += R"METAL(
inline ulong RundUnsignedSqrtU128ToU64(RundU128 value) {
  RundU128 result = RundMakeU128(0ul, 0ul);
  RundU128 bit = RundMakeU128(1ul << 62u, 0ul);
  for (int step = 0; step < 64; ++step) {
    const RundU128 trial = RundAddU128(result, bit);
    if (RundGeU128(value, trial)) {
      value = RundSubU128(value, trial);
      result = RundAddU128(RundShr1U128(result), bit);
    } else {
      result = RundShr1U128(result);
    }
    bit = RundShr2U128(bit);
  }
  return result.lo;
}
)METAL";
}

} // namespace rund::kernel::compute_lowering_detail
