#pragma once

#include <string>

namespace rund::kernel::compute_lowering_detail {

inline void AppendVulkanWideFixedHelpers(std::string &out) {
  out += R"GLSL(
struct RundWide { uint64_t lo; uint64_t hi; };
RundWide RundWideMake(uint64_t lo, uint64_t hi) { return RundWide(lo, hi); }
RundWide RundWideZero() { return RundWideMake(0ul, 0ul); }
RundWide RundWideOne() { return RundWideMake(1ul, 0ul); }
bool RundWideNegative(RundWide value) {
  return (value.hi & 0x8000000000000000ul) != 0ul;
}
RundWide RundWideFrom32(uint value) {
  const uint64_t sign = (value & 0x80000000u) != 0u ? 0xfffffffffffffffful : 0ul;
  return RundWideMake(uint64_t(value) | (sign & 0xffffffff00000000ul), sign);
}
RundWide RundWideFrom64(uint64_t value) {
  return RundWideMake(value, (value & 0x8000000000000000ul) != 0ul
                             ? 0xfffffffffffffffful : 0ul);
}
RundWide RundWideUnsignedLane(RundWide value, uint width) {
  return RundWideMake(width == 32u ? value.lo & 0xfffffffful : value.lo, 0ul);
}
RundWide RundWideAdd(RundWide lhs, RundWide rhs) {
  const uint64_t lo = lhs.lo + rhs.lo;
  return RundWideMake(lo, lhs.hi + rhs.hi + (lo < lhs.lo ? 1ul : 0ul));
}
RundWide RundWideSub(RundWide lhs, RundWide rhs) {
  return RundWideMake(lhs.lo - rhs.lo,
                      lhs.hi - rhs.hi - (lhs.lo < rhs.lo ? 1ul : 0ul));
}
RundWide RundWideNeg(RundWide value) {
  return RundWideAdd(RundWideMake(~value.lo, ~value.hi), RundWideOne());
}
RundWide RundWideAbs(RundWide value) {
  if (RundWideNegative(value)) return RundWideNeg(value);
  return value;
}
RundWide RundWideBool(bool value) {
  if (value) return RundWideOne();
  return RundWideZero();
}
bool RundWideTruthy(RundWide value) { return value.lo != 0ul || value.hi != 0ul; }
RundWide RundWideSelect(bool condition, RundWide when_true, RundWide when_false) {
  if (condition) return when_true;
  return when_false;
}
bool RundWideUnsignedLess(RundWide lhs, RundWide rhs) {
  return lhs.hi < rhs.hi || (lhs.hi == rhs.hi && lhs.lo < rhs.lo);
}
bool RundWideEqual(RundWide lhs, RundWide rhs) {
  return lhs.hi == rhs.hi && lhs.lo == rhs.lo;
}
bool RundWideSignedLess(RundWide lhs, RundWide rhs) {
  const bool lhs_negative = RundWideNegative(lhs);
  const bool rhs_negative = RundWideNegative(rhs);
  return lhs_negative != rhs_negative ? lhs_negative
       : lhs.hi != rhs.hi ? lhs.hi < rhs.hi : lhs.lo < rhs.lo;
}
RundWide RundWideSign(RundWide value) {
  if (RundWideNegative(value)) return RundWideNeg(RundWideOne());
  return RundWideEqual(value, RundWideZero()) ? RundWideZero()
                                              : RundWideOne();
}
RundWide RundWideShl(RundWide value, uint amount) {
  if (amount == 0u) return value;
  if (amount < 64u) {
    return RundWideMake(value.lo << amount,
                        (value.hi << amount) | (value.lo >> (64u - amount)));
  }
  if (amount < 128u) return RundWideMake(0ul, value.lo << (amount - 64u));
  return RundWideZero();
}
RundWide RundWideShrUnsigned(RundWide value, uint amount) {
  if (amount == 0u) return value;
  if (amount < 64u) {
    return RundWideMake((value.lo >> amount) | (value.hi << (64u - amount)),
                        value.hi >> amount);
  }
  if (amount < 128u) return RundWideMake(value.hi >> (amount - 64u), 0ul);
  return RundWideZero();
}
RundWide RundWideShrSigned(RundWide value, uint amount) {
  if (!RundWideNegative(value)) return RundWideShrUnsigned(value, amount);
  const RundWide shifted = RundWideShrUnsigned(
      RundWideMake(~value.lo, ~value.hi), amount);
  return RundWideMake(~shifted.lo, ~shifted.hi);
}
RundWide RundWideAnd(RundWide lhs, RundWide rhs) {
  return RundWideMake(lhs.lo & rhs.lo, lhs.hi & rhs.hi);
}
RundWide RundWideOr(RundWide lhs, RundWide rhs) {
  return RundWideMake(lhs.lo | rhs.lo, lhs.hi | rhs.hi);
}
RundWide RundWideXor(RundWide lhs, RundWide rhs) {
  return RundWideMake(lhs.lo ^ rhs.lo, lhs.hi ^ rhs.hi);
}
RundWide RundWideNot(RundWide value) {
  return RundWideMake(~value.lo, ~value.hi);
}
RundWide RundWideMul64(uint64_t lhs, uint64_t rhs) {
  const uint64_t mask = 0xfffffffful;
  const uint64_t lhs_lo = lhs & mask;
  const uint64_t lhs_hi = lhs >> 32u;
  const uint64_t rhs_lo = rhs & mask;
  const uint64_t rhs_hi = rhs >> 32u;
  const uint64_t p0 = lhs_lo * rhs_lo;
  const uint64_t p1 = lhs_lo * rhs_hi;
  const uint64_t p2 = lhs_hi * rhs_lo;
  const uint64_t p3 = lhs_hi * rhs_hi;
  const uint64_t carry = (p0 >> 32u) + (p1 & mask) + (p2 & mask);
  return RundWideMake((p0 & mask) | (carry << 32u),
                      p3 + (p1 >> 32u) + (p2 >> 32u) + (carry >> 32u));
}
RundWide RundWideMul(RundWide lhs, RundWide rhs) {
  const RundWide base = RundWideMul64(lhs.lo, rhs.lo);
  return RundWideMake(base.lo,
                      base.hi + lhs.lo * rhs.hi + lhs.hi * rhs.lo);
}
RundWide RundWideQuantize(RundWide value, uint source_fraction,
                          uint target_fraction, uint rounding,
                          uint overflow, uint width);
RundWide RundWideQuantizeUnsignedFixedProduct(
    RundWide product, uint source_fraction, uint target_fraction, uint rounding,
    uint overflow, uint width) {
  const uint shift = source_fraction - target_fraction;
  RundWide quotient = RundWideShrUnsigned(product, shift);
  const RundWide remainder = RundWideSub(
      product, RundWideShl(quotient, shift));
  const bool nonzero = !RundWideEqual(remainder, RundWideZero());
  const RundWide halfway = RundWideShl(RundWideOne(), shift - 1u);
  const bool nearest = RundWideUnsignedLess(halfway, remainder) ||
      (RundWideEqual(remainder, halfway) && (quotient.lo & 1ul) != 0ul);
  if ((rounding == 3u && nonzero) || (rounding == 4u && nearest)) {
    quotient = RundWideAdd(quotient, RundWideOne());
  }
  const RundWide maximum = width == 32u
      ? RundWideMake(0xfffffffful, 0ul)
      : RundWideMake(0xfffffffffffffffful, 0ul);
  if (overflow == 1u && RundWideUnsignedLess(maximum, quotient)) {
    quotient = maximum;
  }
  return width == 32u ? RundWideFrom32(uint(quotient.lo))
                      : RundWideFrom64(quotient.lo);
}
RundWide RundWideUnsignedDiv(RundWide numerator, RundWide denominator) {
  RundWide quotient = RundWideZero();
  RundWide remainder = RundWideZero();
  for (int step = 0; step < 128; ++step) {
    const uint bit = 127u - uint(step);
    remainder = RundWideShl(remainder, 1u);
    const uint64_t incoming = bit < 64u ? ((numerator.lo >> bit) & 1ul)
                                        : ((numerator.hi >> (bit - 64u)) & 1ul);
    remainder.lo |= incoming;
    if (!RundWideUnsignedLess(remainder, denominator)) {
      remainder = RundWideSub(remainder, denominator);
      if (bit < 64u) quotient.lo |= 1ul << bit;
      else quotient.hi |= 1ul << (bit - 64u);
    }
  }
  return quotient;
}
RundWide RundWideDivFixed(RundWide lhs, RundWide rhs, uint fraction,
                          uint rounding, uint overflow, uint width) {
  const RundWide zero = RundWideZero();
  if (RundWideEqual(rhs, zero)) {
    if (RundWideEqual(lhs, zero)) return zero;
    const RundWide edge = RundWideNegative(lhs)
        ? RundWideNeg(RundWideShl(RundWideOne(), width - 1u))
        : RundWideSub(RundWideShl(RundWideOne(), width - 1u), RundWideOne());
    return RundWideQuantize(edge, fraction, fraction,
                            rounding, overflow, width);
  }
  const bool negative = RundWideNegative(lhs) != RundWideNegative(rhs);
  const RundWide denominator = RundWideAbs(rhs);
  const RundWide numerator = RundWideShl(RundWideAbs(lhs), fraction);
  RundWide quotient = RundWideUnsignedDiv(numerator, denominator);
  const RundWide remainder = RundWideSub(
      numerator, RundWideMul(quotient, denominator));
  const bool nonzero = !RundWideEqual(remainder, zero);
  const RundWide twice = RundWideShl(remainder, 1u);
  const bool nearest = RundWideUnsignedLess(denominator, twice) ||
      (RundWideEqual(denominator, twice) && (quotient.lo & 1ul) != 0ul);
  if ((rounding == 2u && negative && nonzero) ||
      (rounding == 3u && !negative && nonzero) ||
      (rounding == 4u && nearest)) {
    quotient = RundWideAdd(quotient, RundWideOne());
  }
  const RundWide signed_value = negative ? RundWideNeg(quotient) : quotient;
  return RundWideQuantize(signed_value, fraction, fraction,
                          rounding, overflow, width);
}
RundWide RundWideSqrtFixed(RundWide value, uint fraction,
                           uint rounding, uint overflow, uint width) {
  if (RundWideNegative(value)) return RundWideZero();
  const RundWide radicand = RundWideShl(value, fraction);
  RundWide root = RundWideZero();
  for (int step = 0; step < 64; ++step) {
    const uint bit = 63u - uint(step);
    const RundWide candidate = RundWideOr(
        root, RundWideShl(RundWideOne(), bit));
    const RundWide square = RundWideMul(candidate, candidate);
    if (!RundWideUnsignedLess(radicand, square)) root = candidate;
  }
  return RundWideQuantize(root, fraction, fraction,
                          rounding, overflow, width);
}
RundWide RundWideWrap(RundWide value, uint width) {
  return width == 32u ? RundWideFrom32(uint(value.lo))
                      : RundWideFrom64(value.lo);
}
RundWide RundWideQuantize(RundWide value, uint source_fraction,
                          uint target_fraction, uint rounding,
                          uint overflow, uint width) {
  RundWide low;
  RundWide high;
  if (width == 32u) {
    low = RundWideFrom32(0x80000000u);
    high = RundWideFrom32(0x7fffffffu);
  } else {
    low = RundWideFrom64(0x8000000000000000ul);
    high = RundWideFrom64(0x7ffffffffffffffful);
  }
  RundWide narrowed = value;
  if (target_fraction > source_fraction) {
    const uint shift = target_fraction - source_fraction;
    if (overflow != 2u) {
      const RundWide low_before_scale = RundWideShrSigned(low, shift);
      const RundWide high_before_scale = RundWideShrUnsigned(high, shift);
      if (RundWideSignedLess(value, low_before_scale)) return low;
      if (RundWideSignedLess(high_before_scale, value)) return high;
    }
    narrowed = RundWideShl(value, shift);
  } else if (target_fraction < source_fraction) {
    const uint shift = source_fraction - target_fraction;
    const bool negative = RundWideNegative(value);
    const RundWide magnitude = RundWideAbs(value);
    RundWide quotient = RundWideShrUnsigned(magnitude, shift);
    const RundWide remainder = RundWideSub(magnitude, RundWideShl(quotient, shift));
    const bool nonzero = !RundWideEqual(remainder, RundWideZero());
    const RundWide halfway = RundWideShl(RundWideOne(), shift - 1u);
    const bool nearest = RundWideUnsignedLess(halfway, remainder) ||
        (RundWideEqual(remainder, halfway) && (quotient.lo & 1ul) != 0ul);
    if ((rounding == 2u && negative && nonzero) ||
        (rounding == 3u && !negative && nonzero) ||
        (rounding == 4u && nearest)) {
      quotient = RundWideAdd(quotient, RundWideOne());
    }
    narrowed = negative ? RundWideNeg(quotient) : quotient;
  }
  if (overflow == 2u) return RundWideWrap(narrowed, width);
  if (RundWideSignedLess(narrowed, low)) return low;
  if (RundWideSignedLess(high, narrowed)) return high;
  return RundWideWrap(narrowed, width);
}
)GLSL";
}

} // namespace rund::kernel::compute_lowering_detail
