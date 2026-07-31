#pragma once

#include <string>

namespace rund::kernel::compute_lowering_detail {

inline void AppendVulkanU128Core(std::string &out) {
  out += R"GLSL(
struct RundU128 {
  uint64_t hi;
  uint64_t lo;
};
RundU128 RundMakeU128(uint64_t hi, uint64_t lo) {
  RundU128 value;
  value.hi = hi;
  value.lo = lo;
  return value;
}
bool RundGeU128(RundU128 lhs, RundU128 rhs) {
  return lhs.hi > rhs.hi || (lhs.hi == rhs.hi && lhs.lo >= rhs.lo);
}
RundU128 RundAddU128(RundU128 lhs, RundU128 rhs) {
  const uint64_t lo = lhs.lo + rhs.lo;
  return RundMakeU128(lhs.hi + rhs.hi + (lo < lhs.lo ? 1ul : 0ul), lo);
}
RundU128 RundSubU128(RundU128 lhs, RundU128 rhs) {
  const uint64_t borrow = lhs.lo < rhs.lo ? 1ul : 0ul;
  return RundMakeU128(lhs.hi - rhs.hi - borrow, lhs.lo - rhs.lo);
}
RundU128 RundShr1U128(RundU128 value) {
  return RundMakeU128(value.hi >> 1ul,
                      (value.lo >> 1ul) | (value.hi << 63ul));
}
RundU128 RundShr2U128(RundU128 value) {
  return RundMakeU128(value.hi >> 2ul,
                      (value.lo >> 2ul) | (value.hi << 62ul));
}
RundU128 RundShl1OrU128(RundU128 value, uint64_t bit) {
  return RundMakeU128((value.hi << 1ul) | (value.lo >> 63ul),
                      (value.lo << 1ul) | bit);
}
uint64_t RundBitU128(RundU128 value, int bit) {
  return bit >= 64 ? ((value.hi >> uint64_t(bit - 64)) & 1ul)
                   : ((value.lo >> uint64_t(bit)) & 1ul);
}
RundU128 RundMulWide64(uint64_t lhs, uint64_t rhs) {
  const uint64_t mask = 0xfffffffful;
  const uint64_t lhs0 = lhs & mask;
  const uint64_t lhs1 = lhs >> 32ul;
  const uint64_t rhs0 = rhs & mask;
  const uint64_t rhs1 = rhs >> 32ul;
  const uint64_t product0 = lhs0 * rhs0;
  uint64_t middle = lhs1 * rhs0 + (product0 >> 32ul);
  const uint64_t carry = middle >> 32ul;
  middle = (middle & mask) + lhs0 * rhs1;
  const uint64_t high = lhs1 * rhs1 + carry + (middle >> 32ul);
  const uint64_t low = (middle << 32ul) | (product0 & mask);
  return RundMakeU128(high, low);
}
)GLSL";
}

inline void AppendVulkanU128Division(std::string &out) {
  out += R"GLSL(
uint64_t RundUnsignedDivU128ByU64(RundU128 numerator,
                                  uint64_t denominator) {
  if (denominator == 0ul) { return 0xfffffffffffffffful; }
  uint64_t quotient = 0ul;
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
        quotient |= 1ul << uint64_t(bit);
      }
    }
  }
  return overflow ? 0xfffffffffffffffful : quotient;
}
)GLSL";
}

inline void AppendVulkanU128IntegerSquareRoot(std::string &out) {
  out += R"GLSL(
uint64_t RundUnsignedSqrtU128ToU64(RundU128 value) {
  RundU128 result = RundMakeU128(0ul, 0ul);
  RundU128 bit = RundMakeU128(1ul << 62ul, 0ul);
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
)GLSL";
}

} // namespace rund::kernel::compute_lowering_detail
