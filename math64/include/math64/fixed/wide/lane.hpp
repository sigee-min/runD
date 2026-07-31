#pragma once

#include <math64/fixed/constants.hpp>
#include <math64/simd/arithmetic.hpp>
#include <math64/simd/compare.hpp>
#include <math64/simd/select.hpp>

#include <bit>

namespace rund::math64::detail {

struct alignas(16) I128x2 {
  i128 low = 0;
  i128 high = 0;
};

struct alignas(16) U128x2 {
  u128 low = 0u;
  u128 high = 0u;
};

using Mask128x2 = U128x2;

static_assert(sizeof(I128x2) == 32u);
static_assert(sizeof(U128x2) == 32u);
static_assert(alignof(I128x2) == 16u);
static_assert(alignof(U128x2) == 16u);

[[nodiscard]] inline I128x2 operator+(const I128x2 lhs,
                                      const I128x2 rhs) noexcept {
  return I128x2{lhs.low + rhs.low, lhs.high + rhs.high};
}

[[nodiscard]] inline I128x2 operator-(const I128x2 lhs,
                                      const I128x2 rhs) noexcept {
  return I128x2{lhs.low - rhs.low, lhs.high - rhs.high};
}

[[nodiscard]] inline I128x2 operator-(const I128x2 value) noexcept {
  return I128x2{-value.low, -value.high};
}

[[nodiscard]] inline I128x2 operator*(const I128x2 lhs,
                                      const I128x2 rhs) noexcept {
  return I128x2{lhs.low * rhs.low, lhs.high * rhs.high};
}

[[nodiscard]] inline I128x2 operator>>(const I128x2 value,
                                      const u32 shift) noexcept {
  return I128x2{value.low >> shift, value.high >> shift};
}

[[nodiscard]] inline U128x2 operator+(const U128x2 lhs,
                                      const U128x2 rhs) noexcept {
  return U128x2{lhs.low + rhs.low, lhs.high + rhs.high};
}

[[nodiscard]] inline U128x2 operator-(const U128x2 lhs,
                                      const U128x2 rhs) noexcept {
  return U128x2{lhs.low - rhs.low, lhs.high - rhs.high};
}

[[nodiscard]] inline U128x2 operator*(const U128x2 lhs,
                                      const U128x2 rhs) noexcept {
  return U128x2{lhs.low * rhs.low, lhs.high * rhs.high};
}

[[nodiscard]] inline U128x2 operator&(const U128x2 lhs,
                                      const U128x2 rhs) noexcept {
  return U128x2{lhs.low & rhs.low, lhs.high & rhs.high};
}

[[nodiscard]] inline U128x2 operator|(const U128x2 lhs,
                                      const U128x2 rhs) noexcept {
  return U128x2{lhs.low | rhs.low, lhs.high | rhs.high};
}

[[nodiscard]] inline U128x2 operator^(const U128x2 lhs,
                                      const U128x2 rhs) noexcept {
  return U128x2{lhs.low ^ rhs.low, lhs.high ^ rhs.high};
}

[[nodiscard]] inline U128x2 operator~(const U128x2 value) noexcept {
  return U128x2{~value.low, ~value.high};
}

[[nodiscard]] inline U128x2 operator<<(const U128x2 value,
                                      const u32 shift) noexcept {
  return U128x2{value.low << shift, value.high << shift};
}

[[nodiscard]] inline U128x2 operator<<(const U128x2 value,
                                      const U128x2 shift) noexcept {
  return U128x2{
      value.low << static_cast<u32>(shift.low),
      value.high << static_cast<u32>(shift.high),
  };
}

[[nodiscard]] inline U128x2 operator>>(const U128x2 value,
                                      const u32 shift) noexcept {
  return U128x2{value.low >> shift, value.high >> shift};
}

[[nodiscard]] inline U128x2 operator>>(const U128x2 value,
                                      const U128x2 shift) noexcept {
  return U128x2{
      value.low >> static_cast<u32>(shift.low),
      value.high >> static_cast<u32>(shift.high),
  };
}

inline U128x2 &operator>>=(U128x2 &value, const u32 shift) noexcept {
  value = value >> shift;
  return value;
}

[[nodiscard]] inline I128x2 SplatI128x2(const i128 value) noexcept {
  return I128x2{value, value};
}

[[nodiscard]] inline U128x2 SplatU128x2(const u128 value) noexcept {
  return U128x2{value, value};
}

[[nodiscard]] inline I128x2 Widen128(const simd::I64x value) noexcept {
  return I128x2{
      static_cast<i128>(value[0]),
      static_cast<i128>(value[1]),
  };
}

[[nodiscard]] inline U128x2 Widen128(const simd::U64x value) noexcept {
  return U128x2{
      static_cast<u128>(value[0]),
      static_cast<u128>(value[1]),
  };
}

[[nodiscard]] inline I128x2 Signed128(const U128x2 value) noexcept {
  return std::bit_cast<I128x2>(value);
}

[[nodiscard]] inline U128x2 Unsigned128(const I128x2 value) noexcept {
  return std::bit_cast<U128x2>(value);
}

[[nodiscard]] inline simd::I64x Narrow64(const I128x2 value) noexcept {
  return simd::I64x{
      static_cast<i64>(value.low),
      static_cast<i64>(value.high),
  };
}

[[nodiscard]] inline simd::U64x Narrow64(const U128x2 value) noexcept {
  return simd::U64x{
      static_cast<u64>(value.low),
      static_cast<u64>(value.high),
  };
}

[[nodiscard]] inline simd::Mask64x
NarrowMask(const Mask128x2 value) noexcept {
  return Narrow64(value);
}

[[nodiscard]] inline Mask128x2 Mask128(const I128x2 value) noexcept {
  return Unsigned128(value);
}

[[nodiscard]] inline Mask128x2 Mask128(const U128x2 value) noexcept {
  return value;
}

[[nodiscard]] inline u128 MaskLane(const bool value) noexcept {
  return value ? ~u128{0} : u128{0};
}

[[nodiscard]] inline U128x2
Select128(const Mask128x2 mask, const U128x2 when_true,
          const U128x2 when_false) noexcept {
  return (mask & when_true) | (~mask & when_false);
}

[[nodiscard]] inline I128x2
Select128(const Mask128x2 mask, const I128x2 when_true,
          const I128x2 when_false) noexcept {
  return Signed128(
      Select128(mask, Unsigned128(when_true), Unsigned128(when_false)));
}

[[nodiscard]] inline Mask128x2 Eq128(const U128x2 lhs,
                                     const U128x2 rhs) noexcept {
  return Mask128x2{
      MaskLane(lhs.low == rhs.low),
      MaskLane(lhs.high == rhs.high),
  };
}

[[nodiscard]] inline Mask128x2 Ge128(const U128x2 lhs,
                                     const U128x2 rhs) noexcept {
  return Mask128x2{
      MaskLane(lhs.low >= rhs.low),
      MaskLane(lhs.high >= rhs.high),
  };
}

[[nodiscard]] inline Mask128x2 Gt128(const U128x2 lhs,
                                     const U128x2 rhs) noexcept {
  return Mask128x2{
      MaskLane(lhs.low > rhs.low),
      MaskLane(lhs.high > rhs.high),
  };
}

[[nodiscard]] inline Mask128x2 Lt128(const I128x2 lhs,
                                     const I128x2 rhs) noexcept {
  return Mask128x2{
      MaskLane(lhs.low < rhs.low),
      MaskLane(lhs.high < rhs.high),
  };
}

[[nodiscard]] inline Mask128x2 Gt128(const I128x2 lhs,
                                     const I128x2 rhs) noexcept {
  return Mask128x2{
      MaskLane(lhs.low > rhs.low),
      MaskLane(lhs.high > rhs.high),
  };
}

[[nodiscard]] inline U128x2 Min128(const U128x2 lhs,
                                   const U128x2 rhs) noexcept {
  return Select128(Ge128(lhs, rhs), rhs, lhs);
}

[[nodiscard]] inline simd::I64x
ClampI128x2ToI64x(const I128x2 value) noexcept {
  const I128x2 high = SplatI128x2(static_cast<i128>(FixedMax));
  const I128x2 low = SplatI128x2(static_cast<i128>(FixedMin));
  const I128x2 bounded_high = Select128(Gt128(value, high), high, value);
  const I128x2 bounded =
      Select128(Lt128(bounded_high, low), low, bounded_high);
  return Narrow64(bounded);
}

[[nodiscard]] inline U128x2
AbsMagnitude128(const simd::I64x value) noexcept {
  const I128x2 wide = Widen128(value);
  return Unsigned128(
      Select128(Lt128(wide, SplatI128x2(0)), -wide, wide));
}

[[nodiscard]] inline U128x2
UnsignedDiv128(U128x2 numerator, const U128x2 denominator) noexcept {
  const U128x2 safe_denominator =
      Select128(Eq128(denominator, SplatU128x2(0)), SplatU128x2(1),
                denominator);
  U128x2 quotient = SplatU128x2(0);
  U128x2 remainder = SplatU128x2(0);
  for (int bit = 126; bit >= 0; --bit) {
    remainder =
        (remainder << 1u) | ((numerator >> bit) & SplatU128x2(1));
    const Mask128x2 ge = Ge128(remainder, safe_denominator);
    remainder = Select128(ge, remainder - safe_denominator, remainder);
    quotient =
        quotient |
        Select128(ge, SplatU128x2(u128{1} << bit), SplatU128x2(0));
  }
  return quotient;
}

[[nodiscard]] inline simd::I64x
ClampSignedMagnitude128(const Mask128x2 negative,
                        const U128x2 magnitude) noexcept {
  const U128x2 fixed_min_magnitude = SplatU128x2(u128{1} << 63u);
  const U128x2 fixed_max_magnitude =
      SplatU128x2(static_cast<u64>(FixedMax));
  const U128x2 clamped_positive =
      Select128(Gt128(magnitude, fixed_max_magnitude),
                fixed_max_magnitude, magnitude);
  const I128x2 positive = Signed128(clamped_positive);
  const I128x2 negative_value = -Signed128(
      Select128(Ge128(magnitude, fixed_min_magnitude),
                fixed_min_magnitude, magnitude));
  return Narrow64(Select128(negative, negative_value, positive));
}

}  // namespace rund::math64::detail
