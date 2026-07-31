#pragma once

#include <string_view>

namespace rund::kernel {

inline constexpr std::string_view ReduceWideSource = R"RUND(
struct RundPair { uint lo; uint hi; };
struct RundWide { RUND_REDUCE_U64 lo; RUND_REDUCE_U64 hi; };

RundPair rund_pair_zero() {
  RundPair value;
  value.lo = 0u;
  value.hi = 0u;
  return value;
}

RundPair rund_pair_add(RundPair lhs, uint lo, uint hi) {
  uint sum = lhs.lo + lo;
  uint carry = sum < lhs.lo ? 1u : 0u;
  lhs.lo = sum;
  lhs.hi = lhs.hi + hi + carry;
  return lhs;
}

RundWide rund_wide_make(RUND_REDUCE_U64 lo, RUND_REDUCE_U64 hi) {
  RundWide value;
  value.lo = lo;
  value.hi = hi;
  return value;
}

RundWide rund_wide_zero() {
  return rund_wide_make(RUND_REDUCE_U64(0), RUND_REDUCE_U64(0));
}

RundWide rund_wide_add(RundWide lhs, RundWide rhs) {
  RUND_REDUCE_U64 lo = lhs.lo + rhs.lo;
  return rund_wide_make(
      lo, lhs.hi + rhs.hi + (lo < lhs.lo ? RUND_REDUCE_U64(1)
                                        : RUND_REDUCE_U64(0)));
}

RUND_REDUCE_U64 rund_pair_bits(RundPair value) {
  return RUND_REDUCE_U64(value.lo) |
         (RUND_REDUCE_U64(value.hi) << 32u);
}

RundWide rund_wide_pair_unsigned(RundPair value) {
  return rund_wide_make(rund_pair_bits(value), RUND_REDUCE_U64(0));
}

RundWide rund_wide_pair_signed(RundPair value) {
  RUND_REDUCE_U64 sign = (value.hi & 0x80000000u) != 0u
                             ? ~RUND_REDUCE_U64(0)
                             : RUND_REDUCE_U64(0);
  return rund_wide_make(rund_pair_bits(value), sign);
}

RundWide rund_wide_i32(int value) {
  RUND_REDUCE_U64 sign = value < 0 ? ~RUND_REDUCE_U64(0)
                                   : RUND_REDUCE_U64(0);
  RUND_REDUCE_U64 lo = RUND_REDUCE_U64(uint(value));
  if (value < 0) {
    lo |= ~RUND_REDUCE_U64(0xffffffffu);
  }
  return rund_wide_make(lo, sign);
}

RundWide rund_wide_u32(uint value) {
  return rund_wide_make(RUND_REDUCE_U64(value), RUND_REDUCE_U64(0));
}

RundWide rund_wide_i64(RUND_REDUCE_I64 value) {
  return rund_wide_make(RUND_REDUCE_U64(value),
                        value < RUND_REDUCE_I64(0)
                            ? ~RUND_REDUCE_U64(0)
                            : RUND_REDUCE_U64(0));
}

RundWide rund_wide_u64(RUND_REDUCE_U64 value) {
  return rund_wide_make(value, RUND_REDUCE_U64(0));
}

uint rund_wide_low32(RundWide value) { return uint(value.lo); }

RUND_REDUCE_U64 rund_wide_low64(RundWide value) { return value.lo; }

bool rund_wide_fits_i32(RundWide value) {
  return (value.hi == RUND_REDUCE_U64(0) &&
          value.lo <= RUND_REDUCE_U64(0x7fffffffu)) ||
         (value.hi == ~RUND_REDUCE_U64(0) &&
          value.lo >= ~RUND_REDUCE_U64(0x7fffffffu));
}

bool rund_wide_fits_u32(RundWide value) {
  return value.hi == RUND_REDUCE_U64(0) &&
         value.lo <= RUND_REDUCE_U64(0xffffffffu);
}

bool rund_wide_fits_i64(RundWide value) {
  return (value.hi == RUND_REDUCE_U64(0) &&
          value.lo <= RUND_REDUCE_U64(0x7ffffffffffffffful)) ||
         (value.hi == ~RUND_REDUCE_U64(0) &&
          value.lo >= RUND_REDUCE_U64(0x8000000000000000ul));
}

bool rund_wide_fits_u64(RundWide value) {
  return value.hi == RUND_REDUCE_U64(0);
}
)RUND";

} // namespace rund::kernel
